/* ═══════════════════════════════════════════════════════════════════════════
 * MMU v7  —  High-fidelity Linux MMU simulation | Ultra-low RAM | ESP32-S3
 *
 *  All v6 features retained.  Three new hardware-accurate layers added:
 *
 *  ① Automatic translation  — cpu_ctx_t routes EVERY memory access through
 *     cpu_load_u8/u32 / cpu_store_u8/u32 / cpu_read / cpu_write.
 *     No caller ever calls vm_fault() directly; the translation layer is
 *     transparent, exactly as the hardware MMU operates on every load/store.
 *       v6: caller must call vm_fault() + vm_page_ptr() manually
 *       v7: cpu_load_u8(cpu, addr)  ←  all bookkeeping is implicit
 *
 *  ② Page-fault trap model  — fault_record_t + fault_handler_fn callback.
 *     Every fault is classified (UNMAPPED/GUARD/PERM_R|W|X|PRIV/STACK_OVF/OOM),
 *     logged in cpu->last_fault, and dispatched to a registered handler —
 *     simulating the CPU trapping to the OS via IDT vector #14 (#PF).
 *     A custom handler can mark rec->handled=true to resume execution
 *     (e.g. swapping in a page), otherwise the default handler kills the pid.
 *       v6: printf() + return -SIGSEGV (no handler, no classification)
 *       v7: fault_raise() → typed record → user-replaceable handler → retry/kill
 *
 *  ③ Privilege enforcement  — CPL (0=kernel / 3=user) + MM_USER VMA flag.
 *     cpu_page_ptr() checks cpu->cpl against the VMA's MM_USER bit on every
 *     access, mirroring the x86 U/S bit in each page-table level.
 *     User code touching kernel VMAs raises FAULT_PERM_PRIV and is blocked.
 *       v6: no user/kernel distinction — any code can read any address
 *       v7: CPL=3 + !MM_USER  →  FAULT_PERM_PRIV  →  access denied
 *
 *  RAM budget (unchanged from v6):
 *    PT slab   : 64 × 4 KB  = 256 KB
 *    Frame meta: 256 × 12 B =   3 KB
 *    TLB       : 64×4×12 B  =   3 KB
 *    cpu_ctx_t :             ≈  64 B (stack, per CPU)
 * ═══════════════════════════════════════════════════════════════════════════ */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>

/* ── §0  Config & constants ─────────────────────────────────────────────── */
#ifndef PAGE_SHIFT
# define PAGE_SHIFT      12u
#endif
#ifndef PAGE_SIZE
# define PAGE_SIZE       (1u<<PAGE_SHIFT)
#endif
#ifndef PAGE_MASK
# define PAGE_MASK       (~(PAGE_SIZE-1u))
#endif
#ifndef PAGE_ALIGN
# define PAGE_ALIGN(x)   (((uint32_t)(x)+PAGE_SIZE-1u)&PAGE_MASK)
#endif

#define HUGE_PAGE_SHIFT  22u
#define HUGE_PAGE_SIZE   (1u<<HUGE_PAGE_SHIFT)
#define HUGE_NPAGES      (HUGE_PAGE_SIZE/PAGE_SIZE)

#define PGD_BITS   10u
#define PT_BITS    10u
#define PGD_SIZE   (1u<<PGD_BITS)
#define PT_SIZE    (1u<<PT_BITS)
#define PGD_IDX(va)  ((uint32_t)(va)>>(PAGE_SHIFT+PT_BITS))
#define PT_IDX(va)   (((uint32_t)(va)>>PAGE_SHIFT)&(PT_SIZE-1u))

#ifndef MM_READ
# define MM_READ       (1u<<0)
# define MM_WRITE      (1u<<1)
# define MM_EXEC       (1u<<2)
# define MM_NX         (1u<<3)
# define MM_XIP        (1u<<4)
# define MM_DEVICE     (1u<<5)
# define MM_SHARED     (1u<<6)
# define MM_RO         (1u<<7)
# define MM_GUARD      (1u<<8)
# define MM_ANON       (1u<<9)
# define MM_HUGE       (1u<<10)
# define MM_MERGEABLE  (1u<<11)
# define MM_SEQUENTIAL (1u<<12)
# define MM_RANDOM     (1u<<13)
/* ── v7 new ── */
# define MM_USER       (1u<<14)  /* VMA is accessible from CPL=3 (user mode).
                                  * Kernel VMAs lack this flag; any user-mode
                                  * access to them raises FAULT_PERM_PRIV,
                                  * mirroring the x86 page-table U/S bit.     */
#endif

#ifndef PROT_READ
# define PROT_READ   1
# define PROT_WRITE  2
# define PROT_EXEC   4
# define PROT_NOEXEC 8
# define PROT_RO     16
# define PROT_XIP    32
#endif
#ifndef MAP_PRIVATE
# define MAP_PRIVATE   0x02
# define MAP_SHARED    0x01
# define MAP_ANONYMOUS 0x20
# define MAP_DEVICE    0x100
# define MAP_POPULATE  0x08000
# define MAP_HUGETLB   0x40000
#endif
#ifndef MREMAP_MAYMOVE
# define MREMAP_MAYMOVE 1
# define MREMAP_FIXED   2
#endif
#ifndef MADV_NORMAL
# define MADV_NORMAL     0
# define MADV_RANDOM     1
# define MADV_SEQUENTIAL 2
# define MADV_WILLNEED   3
# define MADV_DONTNEED   4
# define MADV_FREE       5
# define MADV_MERGEABLE  12
#endif
#ifndef MS_SYNC
# define MS_SYNC 4
#endif
#ifndef SIGSEGV
# define SIGSEGV 11
#endif
#ifndef STACK_PAGES
# define STACK_PAGES 16u
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * §1  Frame pool — 256-slot static slab, O(1) alloc/free
 * ═══════════════════════════════════════════════════════════════════════════ */
#define MAX_FRAMES     256u
#define FRAME_NULL     0u
#define FF_DIRTY       (1u<<0)
#define FF_ZRAM        (1u<<1)
#define FF_SD          (1u<<2)

typedef struct {
    uint8_t *data;
    int16_t  ref_count;
    uint8_t  flags;
    uint8_t  sd_block;
} frame_t;   /* 12 B */

static uint8_t  g_zero_page[PAGE_SIZE];
static frame_t  g_frames[MAX_FRAMES];
static uint8_t  g_ffl[MAX_FRAMES];
static uint8_t  g_ffl_head;

static void frame_pool_init(void) {
    static bool done = false;
    if (done) return;
    for (uint32_t i = 0; i < MAX_FRAMES-1u; i++) g_ffl[i] = (uint8_t)(i+1u);
    g_ffl[MAX_FRAMES-1u] = 0xFF; g_ffl_head = 0; done = true;
}
static uint32_t fidx_alloc(int pid) {
    (void)pid;
    if (__builtin_expect(g_ffl_head == 0xFF, 0)) return FRAME_NULL;
    uint32_t idx = g_ffl_head;
    g_ffl_head   = g_ffl[idx];
    frame_t *f   = &g_frames[idx];
    f->ref_count = 1; f->flags = 0; f->sd_block = 0;
    if (!f->data) f->data = (uint8_t *)malloc(PAGE_SIZE);
    if (!f->data) { g_ffl[idx] = g_ffl_head; g_ffl_head = (uint8_t)idx; return FRAME_NULL; }
    memset(f->data, 0, PAGE_SIZE);
    return idx + 1u;
}
static void fidx_release(uint32_t idx) {
    if (!idx || idx > MAX_FRAMES) return;
    frame_t *f = &g_frames[idx-1u];
    if (--f->ref_count > 0) return;
    f->flags = 0;
    g_ffl[idx-1u] = g_ffl_head;
    g_ffl_head    = (uint8_t)(idx-1u);
}
static inline frame_t *fidx_get(uint32_t idx) {
    return (idx && idx <= MAX_FRAMES) ? &g_frames[idx-1u] : NULL;
}
static void fidx_ensure(uint32_t idx) {
    frame_t *f = fidx_get(idx);
    if (!f) return;
    if      (f->flags & FF_SD)   { if (!f->data) f->data = (uint8_t *)calloc(1, PAGE_SIZE); f->flags &= ~FF_SD;   }
    else if (f->flags & FF_ZRAM) { if (!f->data) f->data = (uint8_t *)calloc(1, PAGE_SIZE); f->flags &= ~FF_ZRAM; }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §2  Compact 32-bit PTE  [31:12]=fidx | [11]=P | [10]=D | [9]=A |
 *                          [8]=COW | [7]=HUGE | [6:0]=pf
 * ═══════════════════════════════════════════════════════════════════════════ */
typedef uint32_t pte_t;
#define PTE_FIDX_SHIFT 12u
#define PTE_P    (1u<<11)
#define PTE_D    (1u<<10)
#define PTE_A    (1u<<9)
#define PTE_COW  (1u<<8)
#define PTE_HUGE (1u<<7)
#define PF_R    1u
#define PF_W    2u
#define PF_X    4u
#define PF_NX   8u
#define PF_XIP  16u
#define PF_DEV  32u
#define PF_SHR  64u

#define PTE_FIDX(p)   ((p) >> PTE_FIDX_SHIFT)
#define PTE_FLAGS(p)  ((p) &  0x7Fu)
#define PTE_MAKE(fi,pf,bits) \
    (((uint32_t)(fi) << PTE_FIDX_SHIFT) | (uint32_t)(bits) | (uint32_t)(pf))

static inline uint8_t mmf_pack(uint16_t m) {
    return (uint8_t)(
        ((m & MM_READ)   ? PF_R   : 0) | ((m & MM_WRITE)  ? PF_W   : 0) |
        ((m & MM_EXEC)   ? PF_X   : 0) | ((m & MM_NX)     ? PF_NX  : 0) |
        ((m & MM_XIP)    ? PF_XIP : 0) | ((m & MM_DEVICE) ? PF_DEV : 0) |
        ((m & MM_SHARED) ? PF_SHR : 0));
}
static inline uint16_t mmf_unpack(uint8_t p) {
    return (uint16_t)(
        ((p & PF_R)   ? MM_READ   : 0) | ((p & PF_W)   ? MM_WRITE  : 0) |
        ((p & PF_X)   ? MM_EXEC   : 0) | ((p & PF_NX)  ? MM_NX     : 0) |
        ((p & PF_XIP) ? MM_XIP    : 0) | ((p & PF_DEV) ? MM_DEVICE : 0) |
        ((p & PF_SHR) ? MM_SHARED : 0));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §3  PT slab — 64 static page tables (256 KB total)
 * ═══════════════════════════════════════════════════════════════════════════ */
#define MAX_PT_TABLES 64u
typedef struct { pte_t e[PT_SIZE]; } pt_page_t;
static pt_page_t g_pt_slab[MAX_PT_TABLES];
static uint64_t  g_pt_bmap = 0;

static uint8_t pt_slab_alloc(void) {
    for (int i = 0; i < (int)MAX_PT_TABLES; i++) {
        if (!((g_pt_bmap >> i) & 1)) {
            g_pt_bmap |= (1ULL << i);
            memset(&g_pt_slab[i], 0, sizeof(pt_page_t));
            return (uint8_t)(i + 1u);
        }
    }
    return 0;
}
static void pt_slab_free(uint8_t idx) {
    if (idx && idx <= (uint8_t)MAX_PT_TABLES)
        g_pt_bmap &= ~(1ULL << (idx-1u));
}

#define PGD_HUGE_BIT 0x80u
typedef uint8_t pgd_e_t;

static inline pte_t *pt_walk(pgd_e_t *pgd, uint32_t va, bool alloc) {
    uint32_t gi = PGD_IDX(va);
    if (__builtin_expect(gi >= PGD_SIZE, 0)) return NULL;
    if (pgd[gi] & PGD_HUGE_BIT) return NULL;
    if (!pgd[gi]) {
        if (!alloc) return NULL;
        uint8_t t = pt_slab_alloc(); if (!t) return NULL;
        pgd[gi] = t;
    }
    return &g_pt_slab[pgd[gi]-1u].e[PT_IDX(va)];
}
static void pt_trim(pgd_e_t *pgd, uint32_t start, uint32_t end) {
    uint32_t gi0 = PGD_IDX(start);
    uint32_t gi1 = (end > 0) ? PGD_IDX(end-1u) : 0;
    for (uint32_t gi = gi0; gi <= gi1 && gi < PGD_SIZE; gi++) {
        uint8_t t = pgd[gi] & ~PGD_HUGE_BIT;
        if (!t) continue;
        pt_page_t *pt = &g_pt_slab[t-1u];
        bool empty = true;
        for (int i = 0; i < (int)PT_SIZE && empty; i++)
            if (pt->e[i]) empty = false;
        if (empty) { pt_slab_free(pgd[gi]); pgd[gi] = 0; }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §4  TLB — 64 sets × 4-way pseudo-LRU
 * ═══════════════════════════════════════════════════════════════════════════ */
#define TLB_SETS     64u
#define TLB_WAYS     4u
#define TLB_IDX(vpn) ((uint32_t)(vpn) & (TLB_SETS-1u))

typedef struct {
    uint32_t vpn, ppn;
    uint8_t  asid, pf, tflags, _pad;
} tlb_e_t;   /* 12 B */

#define TF_VALID (1u<<0)
#define TF_HUGE  (1u<<1)

static tlb_e_t g_tlb[TLB_SETS][TLB_WAYS];
static uint8_t g_tlb_plru[TLB_SETS];
static uint8_t g_current_asid = 1;

static inline uint8_t plru_victim(uint8_t st) {
    if (!(st & 4)) return (st & 2) ? 1u : 0u;
    else           return (st & 1) ? 3u : 2u;
}
static inline uint8_t plru_touch(uint8_t st, uint8_t w) {
    if (w < 2) { st = (st & ~4u) | ((w==0)?4u:0u); st = (st & ~2u) | ((w==1)?2u:0u); }
    else       { st = (st & ~4u); st = (st & ~1u) | ((w==3)?1u:0u); }
    return st;
}
static void tlb_flush_asid(uint8_t asid) {
    for (uint32_t s=0;s<TLB_SETS;s++)
        for (uint32_t w=0;w<TLB_WAYS;w++)
            if (g_tlb[s][w].asid==asid) g_tlb[s][w].tflags=0;
}
static void tlb_flush_all(void) {
    memset(g_tlb, 0, sizeof(g_tlb));
    memset(g_tlb_plru, 0, sizeof(g_tlb_plru));
}
static inline tlb_e_t *tlb_lookup(uint32_t vpn, uint8_t asid) {
    uint32_t s = TLB_IDX(vpn);
    tlb_e_t *row = g_tlb[s];
    for (uint32_t w=0; w<TLB_WAYS; w++) {
        tlb_e_t *e = row+w;
        if (!(e->tflags & TF_VALID) || e->asid != asid) continue;
        uint32_t ev = (e->tflags&TF_HUGE)?(e->vpn&~(HUGE_NPAGES-1u)):e->vpn;
        uint32_t qv = (e->tflags&TF_HUGE)?(vpn  &~(HUGE_NPAGES-1u)):vpn;
        if (ev==qv) { g_tlb_plru[s]=plru_touch(g_tlb_plru[s],(uint8_t)w); return e; }
    }
    return NULL;
}
static void tlb_insert(uint32_t vpn, uint32_t ppn, uint8_t pf,
                        uint8_t asid, bool huge) {
    uint32_t s = TLB_IDX(vpn);
    uint8_t  w = plru_victim(g_tlb_plru[s]);
    g_tlb[s][w] = (tlb_e_t){ vpn,ppn,asid,pf,
                              (uint8_t)(TF_VALID|(huge?TF_HUGE:0)),0 };
    g_tlb_plru[s] = plru_touch(g_tlb_plru[s], w);
}
static uint8_t g_asid_next = 1;
static uint8_t vm_alloc_asid(void) {
    uint8_t a=g_asid_next++;
    if (!g_asid_next) g_asid_next=1;
    return a;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §5  VMA — sorted inline array[32], binary search O(log N)
 * ═══════════════════════════════════════════════════════════════════════════ */
#define MAX_VMAS 32u
typedef struct {
    uint32_t start, end, offset;
    uint16_t mm_flags;
    int16_t  fd;
    uint8_t  map_flags, madv, _pad[2];
    char     label[16];
} vma_t;   /* 36 B */

static vma_t *vma_find(vma_t *arr, uint8_t cnt, uint32_t addr) {
    int lo=0, hi=(int)cnt-1;
    while (lo<=hi) {
        int mid=(lo+hi)>>1;
        if      (addr < arr[mid].start) hi=mid-1;
        else if (addr >= arr[mid].end)  lo=mid+1;
        else return &arr[mid];
    }
    return NULL;
}
static void vma_insert_sorted(vma_t *arr, uint8_t *cnt, const vma_t *v) {
    if (*cnt>=MAX_VMAS) return;
    int i=(int)*cnt;
    while (i>0 && arr[i-1].start>v->start) { arr[i]=arr[i-1]; i--; }
    arr[i]=*v; (*cnt)++;
}
static void vma_remove_range(vma_t *arr, uint8_t *cnt, uint32_t s, uint32_t e) {
    uint8_t j=0;
    for (uint8_t i=0;i<*cnt;i++)
        if (arr[i].end<=s || arr[i].start>=e) arr[j++]=arr[i];
    *cnt=j;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §6  vm_space_t  (≈ 6.4 KB per process, all inline)
 * ═══════════════════════════════════════════════════════════════════════════ */
typedef struct {
    pgd_e_t  pgd[PGD_SIZE];
    uint32_t huge_pte[PGD_SIZE];
    vma_t    vmas[MAX_VMAS];
    uint8_t  vma_cnt, asid;
    uint32_t brk, stack_top, mmap_base, aslr_seed;
    uint32_t rss, vsz;
} vm_space_t;

static uint32_t aslr_rand(uint32_t *s) {
    *s = (*s)*1664525u+1013904223u;
    return (*s >> PAGE_SHIFT) & 0x3FFu;
}
static vm_space_t *vm_create(void) {
    frame_pool_init();
    vm_space_t *vm = calloc(1, sizeof(vm_space_t));
    if (!vm) return NULL;
    vm->brk       = 0x08000000u;
    vm->stack_top = 0xC0000000u;
    vm->aslr_seed = (uint32_t)(uintptr_t)vm ^ 0xDEADBEEFu;
    vm->mmap_base = 0x40000000u + aslr_rand(&vm->aslr_seed)*PAGE_SIZE;
    vm->asid      = vm_alloc_asid();
    return vm;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §A  Fault model  ← NEW in v7
 *
 *  Problem v6 had:
 *    • Faults reported only via printf + return -SIGSEGV
 *    • No classification, no handler table, no retry mechanism
 *    • Callers had to inspect return codes themselves
 *
 *  v7 solution:
 *    • fault_type_t  — typed classification of every fault scenario
 *    • fault_record_t — structured record (addr, pid, CPL, R/W/X)
 *    • fault_handler_fn — user-replaceable callback, mirroring OS IDT #PF
 *    • fault_raise()  — single call point that records + dispatches
 *    • rec->handled=true  — handler can resolve the fault (e.g. swap-in)
 *      and cpu_page_ptr will retry once, exactly like OS returning from #PF
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef enum {
    FAULT_NONE       = 0,
    FAULT_UNMAPPED,      /* no VMA at faulting address                        */
    FAULT_GUARD,         /* hit guard page                                    */
    FAULT_STACK_OVF,     /* stack overflow into guard                         */
    FAULT_PERM_R,        /* read attempt on non-readable page                 */
    FAULT_PERM_W,        /* write to read-only / write-protected page         */
    FAULT_PERM_X,        /* execute attempt on NX page                        */
    FAULT_PERM_PRIV,     /* CPL=3 (user) accessing kernel VMA (!MM_USER)      */
    FAULT_OOM,           /* physical frame allocator exhausted                */
} fault_type_t;

typedef struct {
    fault_type_t type;
    uint32_t     addr;       /* faulting virtual address                      */
    int          pid;
    uint8_t      cpl;        /* CPL at time of fault (0=kernel, 3=user)       */
    bool         is_write;
    bool         is_exec;
    bool         handled;    /* handler sets true to request retry            */
} fault_record_t;

/* Forward-declare cpu_ctx so the handler typedef can reference it */
typedef struct cpu_ctx cpu_ctx_t;

/* Fault handler signature — mirroring OS page-fault ISR.
 * Set rec->handled = true to signal the fault was resolved and
 * cpu_page_ptr() should retry the page walk.                                */
typedef void (*fault_handler_fn)(cpu_ctx_t *cpu, fault_record_t *rec);

/* ── cpu_ctx_t ─────────────────────────────────────────────────────────────
 *  One per simulated CPU / thread.  Holds everything the MMU hardware
 *  carries implicitly: the address space (vm), privilege level (cpl),
 *  the pending fault record, and the fault handler pointer.
 *
 *  "Automatic translation" means: caller writes
 *        cpu_store_u32(cpu, addr, value);
 *  and never thinks about page-tables, TLB, or faults — identical in
 *  interface to how a CPU's store instruction works.                        */
struct cpu_ctx {
    vm_space_t      *vm;
    int              pid;
    uint8_t          cpl;           /* 0 = kernel ring, 3 = user ring        */
    bool             killed;        /* set by default handler on fatal fault  */
    fault_record_t   last_fault;    /* most recent fault (FAULT_NONE if none) */
    fault_handler_fn fault_handler; /* NULL → default_fault_handler          */
};

/* ── fault names (for default handler output) ─────────────────────────── */
static const char * const g_fault_names[] = {
    "NONE", "UNMAPPED", "GUARD", "STACK_OVF",
    "PERM_R", "PERM_W", "PERM_X", "PERM_PRIV", "OOM"
};

/* ── default_fault_handler ─────────────────────────────────────────────── *
 *  Simulates the OS's default SIGSEGV delivery: print and kill.
 *  Replace cpu->fault_handler with a custom function to intercept.         */
static void default_fault_handler(cpu_ctx_t *cpu, fault_record_t *rec) {
    const char *name = (rec->type < (fault_type_t)(sizeof g_fault_names/sizeof*g_fault_names))
                       ? g_fault_names[rec->type] : "UNKNOWN";
    fprintf(stderr,
        "[#PF] %-12s  pid=%-3d  cpl=%d  addr=0x%08X  %s%s\n",
        name, rec->pid, rec->cpl, rec->addr,
        rec->is_write ? "WRITE" : "READ",
        rec->is_exec  ? "+EXEC" : "");
    cpu->killed = true;
    /* rec->handled remains false → caller returns NULL / 0                 */
}

/* ── fault_raise ────────────────────────────────────────────────────────── *
 *  Single call site for every fault.  Records the fault, dispatches the
 *  handler, and returns:
 *    0           if handler resolved it (rec->handled = true)
 *    -SIGSEGV    for permission / unmapped faults
 *    -ENOMEM     for OOM
 *  Mirrors the CPU's trap-to-OS-and-return path.                           */
static int fault_raise(cpu_ctx_t *cpu, uint32_t addr, fault_type_t ft,
                       bool wr, bool ex) {
    fault_record_t *r = &cpu->last_fault;
    r->type     = ft;
    r->addr     = addr;
    r->pid      = cpu->pid;
    r->cpl      = cpu->cpl;
    r->is_write = wr;
    r->is_exec  = ex;
    r->handled  = false;
    fault_handler_fn h = cpu->fault_handler ? cpu->fault_handler
                                             : default_fault_handler;
    h(cpu, r);
    if (r->handled) return 0;
    return (ft == FAULT_OOM) ? -ENOMEM : -SIGSEGV;
}

/* ── cpu_ctx helpers ────────────────────────────────────────────────────── */
static inline void cpu_init(cpu_ctx_t *cpu, vm_space_t *vm, int pid, uint8_t cpl) {
    cpu->vm            = vm;
    cpu->pid           = pid;
    cpu->cpl           = cpl;
    cpu->killed        = false;
    cpu->fault_handler = NULL;   /* use default */
    cpu->last_fault    = (fault_record_t){ FAULT_NONE, 0, pid, cpl, false, false, false };
}
static inline bool cpu_ok(const cpu_ctx_t *cpu) { return !cpu->killed; }

/* ── privilege helpers ─────────────────────────────────────────────────── */
static inline bool vma_accessible(const vma_t *v, uint8_t cpl) {
    /* Kernel (CPL=0) can always access; user (CPL=3) needs MM_USER          */
    return (cpl == 0) || ((v->mm_flags & MM_USER) != 0);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §7  Low-level PT / VMA accessors
 * ═══════════════════════════════════════════════════════════════════════════ */
static inline pte_t *pte_ptr(vm_space_t *vm, uint32_t va, bool alloc) {
    if (vm->pgd[PGD_IDX(va)] & PGD_HUGE_BIT) return NULL;
    return pt_walk(vm->pgd, va, alloc);
}
static void vm_add_vma(vm_space_t *vm, uint32_t s, uint32_t e,
                        uint16_t flags, int mf, int fd,
                        uint32_t off, const char *lbl) {
    vma_t v={0};
    v.start=s; v.end=e; v.mm_flags=flags;
    v.map_flags=(uint8_t)mf; v.fd=(int16_t)fd; v.offset=off;
    if (lbl) strncpy(v.label, lbl, 15);
    vma_insert_sorted(vm->vmas, &vm->vma_cnt, &v);
    vm->vsz += (e-s)>>PAGE_SHIFT;
}
static inline vma_t *vm_vma_find(vm_space_t *vm, uint32_t addr) {
    return vma_find(vm->vmas, vm->vma_cnt, addr);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §8  mprotect
 * ═══════════════════════════════════════════════════════════════════════════ */
static int vm_mprotect(vm_space_t *vm, uint32_t addr, uint32_t len,
                        int prot, int pid) {
    uint32_t end=addr+len;
    for (uint8_t i=0; i<vm->vma_cnt; i++) {
        vma_t *v=&vm->vmas[i];
        if (v->end<=addr || v->start>=end) continue;
        if ((prot&PROT_EXEC) && (v->mm_flags&MM_NX))  { printf("[MM] mprotect DENIED NX  pid=%d\n",pid);    return -EACCES; }
        if (v->mm_flags & MM_GUARD)                    { printf("[MM] mprotect DENIED guard pid=%d\n",pid);  return -EACCES; }
        if (!(prot&PROT_READ) && (v->mm_flags&MM_XIP)) { printf("[MM] mprotect DENIED XIP pid=%d\n",pid);   return -EACCES; }
        v->mm_flags &= ~(uint16_t)(MM_READ|MM_WRITE|MM_EXEC);
        if (prot&PROT_READ)  v->mm_flags |= MM_READ;
        if (prot&PROT_WRITE) v->mm_flags |= MM_WRITE;
        if (prot&PROT_EXEC)  v->mm_flags |= MM_EXEC;
        uint8_t npf = mmf_pack(v->mm_flags);
        for (uint32_t a=v->start; a<v->end; a+=PAGE_SIZE) {
            pte_t *p=pte_ptr(vm,a,false);
            if (p && *p) *p=(*p & ~0x7Fu)|npf;
        }
        tlb_flush_asid(vm->asid);
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §9  madvise
 * ═══════════════════════════════════════════════════════════════════════════ */
static int vm_madvise(vm_space_t *vm, uint32_t addr, uint32_t len, int advice) {
    uint32_t end=addr+len;
    for (uint8_t i=0; i<vm->vma_cnt; i++) {
        vma_t *v=&vm->vmas[i];
        if (v->end<=addr || v->start>=end) continue;
        v->madv=(uint8_t)advice;
        switch (advice) {
        case MADV_DONTNEED:
        case MADV_FREE:
            for (uint32_t a=v->start; a<v->end; a+=PAGE_SIZE) {
                pte_t *p=pte_ptr(vm,a,false);
                if (!p || !(*p & PTE_P)) continue;
                frame_t *f=fidx_get(PTE_FIDX(*p));
                if (f && f->ref_count==1) { fidx_release(PTE_FIDX(*p)); *p=0; vm->rss--; }
            }
            pt_trim(vm->pgd, v->start, v->end);
            break;
        case MADV_WILLNEED:
            for (uint32_t a=v->start; a<v->end; a+=PAGE_SIZE) {
                pte_t *p=pte_ptr(vm,a,true);
                if (!p || (*p & PTE_P)) continue;
                uint32_t fi=fidx_alloc(0); if (!fi) goto willneed_oom;
                *p=PTE_MAKE(fi, mmf_pack(v->mm_flags), PTE_P|PTE_A);
                vm->rss++;
            }
            willneed_oom: break;
        case MADV_SEQUENTIAL:
            v->mm_flags=(v->mm_flags&~(uint16_t)MM_RANDOM)|MM_SEQUENTIAL; break;
        case MADV_RANDOM:
            v->mm_flags=(v->mm_flags&~(uint16_t)MM_SEQUENTIAL)|MM_RANDOM; break;
        }
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §10  msync
 * ═══════════════════════════════════════════════════════════════════════════ */
static int vm_msync(vm_space_t *vm, uint32_t addr, uint32_t len, int flags) {
    uint32_t end=addr+len;
    for (uint8_t i=0; i<vm->vma_cnt; i++) {
        vma_t *v=&vm->vmas[i];
        if (v->end<=addr || v->start>=end) continue;
        for (uint32_t a=v->start; a<v->end; a+=PAGE_SIZE) {
            pte_t *p=pte_ptr(vm,a,false);
            if (!p || !(*p & PTE_D)) continue;
            if (flags & MS_SYNC) *p &= ~PTE_D;
        }
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §11  mremap
 * ═══════════════════════════════════════════════════════════════════════════ */
static uint32_t vm_mremap(vm_space_t *vm, uint32_t old_addr, uint32_t old_sz,
                            uint32_t new_sz, int flags, uint32_t hint) {
    (void)old_sz;
    vma_t *v=vm_vma_find(vm,old_addr);
    if (!v) return (uint32_t)-EINVAL;
    if (new_sz <= (v->end-v->start)) { v->end=v->start+PAGE_ALIGN(new_sz); return old_addr; }
    uint32_t new_end=v->start+PAGE_ALIGN(new_sz);
    bool conflict=false;
    for (uint8_t i=0; i<vm->vma_cnt; i++) {
        vma_t *o=&vm->vmas[i];
        if (o!=v && o->start<new_end && o->end>v->end) { conflict=true; break; }
    }
    if (!conflict) { v->end=new_end; return old_addr; }
    if (!(flags & MREMAP_MAYMOVE)) return (uint32_t)-ENOMEM;
    uint32_t new_addr=(flags&MREMAP_FIXED)?hint:vm->mmap_base;
    if (!(flags&MREMAP_FIXED)) vm->mmap_base=new_addr+PAGE_ALIGN(new_sz);
    uint32_t old_end=v->end;
    for (uint32_t a=v->start; a<old_end; a+=PAGE_SIZE) {
        pte_t *src=pte_ptr(vm,a,false); if (!src||!*src) continue;
        pte_t *dst=pte_ptr(vm,new_addr+(a-v->start),true); if (!dst) break;
        *dst=*src; *src=0;
    }
    vma_t nv=*v; nv.start=new_addr; nv.end=new_addr+PAGE_ALIGN(new_sz);
    vma_remove_range(vm->vmas, &vm->vma_cnt, v->start, old_end);
    vma_insert_sorted(vm->vmas, &vm->vma_cnt, &nv);
    pt_trim(vm->pgd, v->start, old_end);
    tlb_flush_asid(vm->asid);
    return new_addr;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §12  vm_fault — 9 scenarios + v7 privilege enforcement
 *
 *  v6 signature: vm_fault(vm_space_t*, uint32_t addr, int write, int exec, int pid)
 *  v7 signature: vm_fault(cpu_ctx_t*, uint32_t addr, int write, int exec)
 *
 *  New in v7:
 *   A) Privilege check immediately after VMA lookup:
 *        if (cpu->cpl == 3 && !(vma->mm_flags & MM_USER))
 *            → fault_raise(FAULT_PERM_PRIV)
 *      This enforces the hardware U/S page-table bit in software.
 *
 *   B) All printf+return-SIGSEGV paths replaced by fault_raise(), which
 *      records a structured fault_record_t and calls the registered handler.
 *      If the handler sets rec->handled=true, vm_fault returns 0 (resolved).
 *
 *   C) OOM path now raises FAULT_OOM instead of silently returning -ENOMEM,
 *      giving the handler a chance to free pages and retry.
 * ═══════════════════════════════════════════════════════════════════════════ */
static int vm_fault(cpu_ctx_t *cpu, uint32_t addr, int write, int exec_f) {
    vm_space_t *vm  = cpu->vm;
    int         pid = cpu->pid;
    uint32_t    vpn = addr >> PAGE_SHIFT;

    /* ─ ① TLB fast path ─────────────────────────────────────────────────── */
    tlb_e_t *te = tlb_lookup(vpn, vm->asid);
    if (__builtin_expect(te != NULL, 1)) {
        if (exec_f && !(te->pf & PF_X) && !(te->pf & PF_DEV))
            return fault_raise(cpu, addr, FAULT_PERM_X, false, true);
        if (write  && !(te->pf & PF_W))
            return fault_raise(cpu, addr, FAULT_PERM_W, true, false);
        return 0;
    }

    /* ─ ② PT walk ─────────────────────────────────────────────────────────── */
    pte_t *pp = pte_ptr(vm, addr, false);
    pte_t  pv = pp ? *pp : 0;

    if (!pp || !(pv & PTE_P)) {
        /* ─ ③ Demand fault: find VMA ──────────────────────────────────────── */
        vma_t *vma = vm_vma_find(vm, addr);
        if (!vma) {
            uint32_t guard = vm->stack_top - (STACK_PAGES+1u)*PAGE_SIZE;
            if (addr>=guard && addr<guard+PAGE_SIZE)
                return fault_raise(cpu, addr, FAULT_STACK_OVF, (bool)write, false);
            return fault_raise(cpu, addr, FAULT_UNMAPPED, (bool)write, (bool)exec_f);
        }
        if (vma->mm_flags & MM_GUARD)
            return fault_raise(cpu, addr, FAULT_GUARD, (bool)write, false);

        /* ─ ③-A  Privilege check (v7)  ─────────────────────────────────────
         *  Real hardware: U/S bit in PGD/PT checked by MMU on every walk.
         *  Simulator: we check VMA flag MM_USER against cpu->cpl here,
         *  immediately after VMA is found, before any frame is allocated.   */
        if (!vma_accessible(vma, cpu->cpl))
            return fault_raise(cpu, addr, FAULT_PERM_PRIV, (bool)write, false);

        uint8_t pf = mmf_pack(vma->mm_flags);
        pp = pte_ptr(vm, addr, true);
        if (!pp) return fault_raise(cpu, addr, FAULT_OOM, (bool)write, false);

        /* ─ ④ MMIO device mapping ──────────────────────────────────────────── */
        if (vma->mm_flags & MM_DEVICE) {
            uint32_t xppn = (vma->offset + (addr & PAGE_MASK)) >> PAGE_SHIFT;
            *pp = PTE_MAKE(xppn, pf, PTE_P);
            tlb_insert(vpn, xppn, pf, vm->asid, false);
            return 0;
        }
        /* ─ ⑤ XIP — execute / read direct from flash ──────────────────────── */
        if (vma->mm_flags & MM_XIP) {
            uint32_t xppn = (vma->offset + (addr & PAGE_MASK)) >> PAGE_SHIFT;
            bool huge = (vma->mm_flags & MM_HUGE) != 0;
            *pp = PTE_MAKE(xppn, pf, PTE_P|(huge?PTE_HUGE:0));
            tlb_insert(vpn, xppn, pf, vm->asid, huge);
            return 0;
        }
        /* ─ ⑥ Allocate physical frame ──────────────────────────────────────── */
        uint32_t fi = fidx_alloc(pid);
        if (!fi) return fault_raise(cpu, addr, FAULT_OOM, (bool)write, false);
        pv = PTE_MAKE(fi, pf, PTE_P|PTE_A);
        *pp = pv; vm->rss++;

        /* ─ ⑦ Sequential read-ahead: prefetch 4 pages ─────────────────────── */
        if (vma->mm_flags & MM_SEQUENTIAL) {
            for (int k=1; k<=4; k++) {
                uint32_t na=(addr+(uint32_t)k*PAGE_SIZE)&PAGE_MASK;
                if (na>=vma->end) break;
                pte_t *np=pte_ptr(vm,na,true);
                if (np && !(*np & PTE_P)) {
                    uint32_t nfi=fidx_alloc(pid);
                    if (nfi) { *np=PTE_MAKE(nfi,pf,PTE_P|PTE_A); vm->rss++; }
                }
            }
        }
        tlb_insert(vpn, PTE_FIDX(pv), pf, vm->asid, false);
        return 0;
    }

    /* Page present, not in TLB — re-populate; check permissions */
    uint8_t pf = (uint8_t)PTE_FLAGS(pv);

    /* Privilege re-check on present pages (VMA may have changed via mprotect) */
    {
        vma_t *vma = vm_vma_find(vm, addr);
        if (vma && !vma_accessible(vma, cpu->cpl))
            return fault_raise(cpu, addr, FAULT_PERM_PRIV, (bool)write, false);
    }

    if (exec_f  && (pf & PF_NX))             return fault_raise(cpu,addr,FAULT_PERM_X,false,true);
    if (write   && (pv & PTE_COW))           goto cow;
    if (write   && !(pf & PF_W))             return fault_raise(cpu,addr,FAULT_PERM_W,true,false);
    if (write   && (pf & (PF_XIP|PF_DEV)))   return fault_raise(cpu,addr,FAULT_PERM_W,true,false);
    if (!write  && !(pf & PF_R))             return fault_raise(cpu,addr,FAULT_PERM_R,false,false);

    fidx_ensure(PTE_FIDX(pv));
    *pp = pv | PTE_A;
    tlb_insert(vpn, PTE_FIDX(pv), pf, vm->asid, (pv & PTE_HUGE)!=0);
    return 0;

    /* ─ ⑧ COW break ─────────────────────────────────────────────────────── */
cow:;
    {
        uint32_t old_fi = PTE_FIDX(pv);
        frame_t *of = fidx_get(old_fi);
        uint32_t new_fi = fidx_alloc(pid);
        if (!new_fi) return fault_raise(cpu, addr, FAULT_OOM, true, false);
        frame_t *nf = fidx_get(new_fi);
        if (of && of->data && nf && nf->data)
            memcpy(nf->data, of->data, PAGE_SIZE);
        if (of && --of->ref_count <= 0) fidx_release(old_fi);
        *pp = PTE_MAKE(new_fi, pf|PF_W, PTE_P|PTE_A|PTE_D);
        tlb_flush_asid(vm->asid);
        return 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §13  Automatic translation layer  ← NEW in v7
 *
 *  Problem v6 had:
 *    • vm_page_ptr(vm, addr, pid, wr) returns NULL on fault — caller must
 *      handle NULL everywhere; fault() must be called manually before R/W.
 *    • No privilege check at the R/W call site.
 *
 *  v7 solution — cpu_page_ptr(cpu, addr, wr):
 *    • AUTOMATICALLY calls vm_fault() on TLB miss (transparent to caller).
 *    • Enforces read/write permission via fault_raise() on violation.
 *    • Retries once after a "handled" fault (handler resolved the condition).
 *    • All higher-level functions (cpu_load_u8, cpu_store_u32, cpu_read,
 *      cpu_write) are built on top — callers just read/write addresses.
 *
 *  Legacy vm_rb/vm_wb/vm_r8/vm_w8/vm_r32 are kept as thin wrappers that
 *  create a temporary kernel (CPL=0) cpu_ctx, preserving backward compat.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* cpu_page_ptr — resolves VA to host pointer with automatic translation.
 *
 *  On TLB miss → vm_fault() called automatically (no explicit call needed).
 *  If fault is handled (rec->handled=true) → retries once.
 *  On unresolvable fault → cpu->killed set, returns NULL.                   */
static uint8_t *cpu_page_ptr(cpu_ctx_t *cpu, uint32_t addr, bool wr) {
    uint32_t vpn = addr >> PAGE_SHIFT;
    tlb_e_t *te  = tlb_lookup(vpn, cpu->vm->asid);
    if (__builtin_expect(!te, 0)) {
        int r = vm_fault(cpu, addr, wr?1:0, 0);
        if (r < 0) return NULL;                        /* fault not resolved */
        te = tlb_lookup(vpn, cpu->vm->asid);
        if (__builtin_expect(!te, 0)) return NULL;
    }
    /* ── Privilege check on TLB hit ─────────────────────────────────────
     *  Real hardware stores U/S bit per TLB entry and checks it on every
     *  access.  We check the VMA here to enforce the same policy even when
     *  the TLB was populated by a kernel-mode access to the same page.     */
    if (cpu->cpl == 3) {
        vma_t *vma = vm_vma_find(cpu->vm, addr);
        if (!vma || !vma_accessible(vma, cpu->cpl)) {
            fault_raise(cpu, addr, FAULT_PERM_PRIV, wr, false);
            return NULL;
        }
    }
    /* ── Permission enforcement — always checked, no bypass ───────────── */
    if (wr  && !(te->pf & PF_W)) {
        fault_raise(cpu, addr, FAULT_PERM_W, true, false); return NULL; }
    if (!wr && !(te->pf & PF_R)) {
        fault_raise(cpu, addr, FAULT_PERM_R, false, false); return NULL; }
    if (te->pf & (PF_XIP|PF_DEV)) return g_zero_page;   /* flash/MMIO sim   */
    frame_t *f = fidx_get(te->ppn);
    if (__builtin_expect(!f || !f->data, 0)) return NULL;
    if (wr) {
        pte_t *p = pte_ptr(cpu->vm, addr, false);
        if (p) *p |= PTE_D;
        f->flags |= FF_DIRTY;
    }
    return f->data;
}

/* ── cpu_load / cpu_store ───────────────────────────────────────────────── *
 *  These are the user-facing "instruction-level" API.  A call to
 *  cpu_load_u32(cpu, addr) is semantically equivalent to executing a
 *  32-bit LOAD instruction on a CPU with this MMU — the translation,
 *  fault handling, and permission checks are all invisible.               */

static inline uint8_t cpu_load_u8(cpu_ctx_t *cpu, uint32_t addr) {
    uint8_t *base = cpu_page_ptr(cpu, addr, false);
    return base ? base[addr & (PAGE_SIZE-1u)] : 0u;
}
static inline void cpu_store_u8(cpu_ctx_t *cpu, uint32_t addr, uint8_t val) {
    uint8_t *base = cpu_page_ptr(cpu, addr, true);
    if (base && base != g_zero_page) base[addr & (PAGE_SIZE-1u)] = val;
}

static inline uint32_t cpu_load_u32(cpu_ctx_t *cpu, uint32_t addr) {
    /* aligned fast path: everything in one page → single memcpy */
    if (__builtin_expect((addr&3u)==0 && (addr&(PAGE_SIZE-1u))<=PAGE_SIZE-4u, 1)) {
        uint8_t *base = cpu_page_ptr(cpu, addr, false);
        if (base) { uint32_t v; memcpy(&v, base+(addr&(PAGE_SIZE-1u)), 4); return v; }
        return 0u;
    }
    /* unaligned / cross-page: byte-wise */
    uint32_t v=0u;
    for (int i=0;i<4;i++) v |= (uint32_t)cpu_load_u8(cpu, addr+i) << (i*8);
    return v;
}
static inline void cpu_store_u32(cpu_ctx_t *cpu, uint32_t addr, uint32_t val) {
    if (__builtin_expect((addr&3u)==0 && (addr&(PAGE_SIZE-1u))<=PAGE_SIZE-4u, 1)) {
        uint8_t *base = cpu_page_ptr(cpu, addr, true);
        if (base && base != g_zero_page)
            memcpy(base+(addr&(PAGE_SIZE-1u)), &val, 4);
        return;
    }
    for (int i=0;i<4;i++) cpu_store_u8(cpu, addr+i, (uint8_t)(val>>(i*8)));
}

/* Bulk read — page-granular, no per-byte fault loop */
static void cpu_read(cpu_ctx_t *cpu, uint32_t addr, void *dst, uint32_t n) {
    uint8_t *d = (uint8_t *)dst;
    while (n) {
        uint32_t off   = addr & (PAGE_SIZE-1u);
        uint32_t chunk = PAGE_SIZE - off; if (chunk > n) chunk = n;
        uint8_t *base  = cpu_page_ptr(cpu, addr, false);
        if (base) memcpy(d, base+off, chunk);
        else      memset(d, 0,        chunk);
        addr += chunk; d += chunk; n -= chunk;
    }
}
/* Bulk write — page-granular */
static void cpu_write(cpu_ctx_t *cpu, uint32_t addr, const void *src, uint32_t n) {
    const uint8_t *s = (const uint8_t *)src;
    while (n) {
        uint32_t off   = addr & (PAGE_SIZE-1u);
        uint32_t chunk = PAGE_SIZE - off; if (chunk > n) chunk = n;
        uint8_t *base  = cpu_page_ptr(cpu, addr, true);
        if (base && base != g_zero_page) memcpy(base+off, s, chunk);
        addr += chunk; s += chunk; n -= chunk;
    }
}

/* ── Legacy wrappers (backward compat) ─────────────────────────────────── *
 *  Create a temporary kernel CPU context (CPL=0) so old callers that
 *  pass (vm, pid) still work without modification.                         */
static uint8_t *vm_page_ptr(vm_space_t *vm, uint32_t addr, int pid, bool wr) {
    cpu_ctx_t tmp; cpu_init(&tmp, vm, pid, 0);
    return cpu_page_ptr(&tmp, addr, wr);
}
static uint8_t  vm_r8 (vm_space_t *vm, uint32_t a, int pid) {
    cpu_ctx_t t; cpu_init(&t,vm,pid,0); return cpu_load_u8(&t,a); }
static void     vm_w8 (vm_space_t *vm, uint32_t a, uint8_t v, int pid) {
    cpu_ctx_t t; cpu_init(&t,vm,pid,0); cpu_store_u8(&t,a,v); }
static void     vm_rb (vm_space_t *vm, uint32_t a, void *d, uint32_t n, int pid) {
    cpu_ctx_t t; cpu_init(&t,vm,pid,0); cpu_read(&t,a,d,n); }
static void     vm_wb (vm_space_t *vm, uint32_t a, const void *s, uint32_t n, int pid) {
    cpu_ctx_t t; cpu_init(&t,vm,pid,0); cpu_write(&t,a,s,n); }
static uint32_t vm_r32(vm_space_t *vm, uint32_t a, int pid) {
    cpu_ctx_t t; cpu_init(&t,vm,pid,0); return cpu_load_u32(&t,a); }

/* ═══════════════════════════════════════════════════════════════════════════
 * §14  mmap / munmap  (+ cpu_mmap for user-mode mappings)
 * ═══════════════════════════════════════════════════════════════════════════ */
static uint32_t vm_mmap(vm_space_t *vm, uint32_t hint, uint32_t len,
                         int prot, int mflags, int fd, uint32_t off,
                         uint16_t extra_mmf) {
    uint32_t addr = hint ? hint : vm->mmap_base;
    uint32_t alen = PAGE_ALIGN(len);
    if (!hint) vm->mmap_base = addr + alen;
    uint16_t mmf = extra_mmf;
    if (prot & PROT_READ)   mmf |= MM_READ;
    if (prot & PROT_WRITE)  mmf |= MM_WRITE;
    if (prot & PROT_EXEC)   mmf |= MM_EXEC;
    if (prot & PROT_NOEXEC) mmf |= MM_NX;
    if (prot & PROT_RO)     mmf |= MM_RO;
    if (prot & PROT_XIP)    mmf |= (uint16_t)(MM_XIP|MM_EXEC|MM_READ);
    if (mflags & MAP_SHARED)    mmf |= MM_SHARED;
    if (mflags & MAP_ANONYMOUS) mmf |= MM_ANON;
    if (mflags & MAP_DEVICE)    mmf |= MM_DEVICE;
    if (mflags & MAP_HUGETLB)   mmf |= MM_HUGE;
    const char *lbl = (mflags & MAP_ANONYMOUS) ? "[anon]" : "[file]";
    if (mflags & MAP_DEVICE) lbl = "[mmio]";
    vm_add_vma(vm, addr, addr+alen, mmf, mflags, fd, off, lbl);
    if (mflags & MAP_POPULATE) {
        uint8_t pf = mmf_pack(mmf);
        for (uint32_t a=addr; a<addr+alen; a+=PAGE_SIZE) {
            pte_t *p=pte_ptr(vm,a,true); if (!p) break;
            if (*p & PTE_P) continue;
            if (mmf & (MM_XIP|MM_DEVICE)) {
                uint32_t xppn=(off+(a-addr))>>PAGE_SHIFT;
                *p=PTE_MAKE(xppn,pf,PTE_P);
            } else {
                uint32_t fi=fidx_alloc(0); if (!fi) break;
                *p=PTE_MAKE(fi,pf,PTE_P|PTE_A); vm->rss++;
            }
        }
    }
    return addr;
}

/* cpu_mmap — user-mode aware mmap.
 *  Sets MM_USER automatically when cpu->cpl == 3, so the VMA is marked
 *  as user-accessible and privilege checks in vm_fault pass.
 *  This mirrors the OS kernel setting the U/S bit when mapping user pages. */
static uint32_t cpu_mmap(cpu_ctx_t *cpu, uint32_t hint, uint32_t len,
                           int prot, int mflags, int fd, uint32_t off) {
    uint16_t extra = (cpu->cpl == 3) ? MM_USER : 0;
    return vm_mmap(cpu->vm, hint, len, prot, mflags, fd, off, extra);
}

static void vm_munmap(vm_space_t *vm, uint32_t addr, uint32_t len) {
    uint32_t end=addr+len;
    for (uint32_t a=addr; a<end; a+=PAGE_SIZE) {
        pte_t *p=pte_ptr(vm,a,false); if (!p||!*p) continue;
        uint32_t fi=PTE_FIDX(*p);
        if (fi) { frame_t *f=fidx_get(fi); if (f && --f->ref_count<=0) fidx_release(fi); }
        *p=0; if (vm->rss) vm->rss--;
    }
    vma_remove_range(vm->vmas, &vm->vma_cnt, addr, end);
    pt_trim(vm->pgd, addr, end);
    vm->vsz -= (end-addr)>>PAGE_SHIFT;
    tlb_flush_asid(vm->asid);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §15  COW fork clone
 * ═══════════════════════════════════════════════════════════════════════════ */
static vm_space_t *vm_clone_cow(vm_space_t *par) {
    vm_space_t *ch = calloc(1, sizeof(vm_space_t));
    if (!ch) return NULL;
    memcpy(ch->vmas, par->vmas, par->vma_cnt*sizeof(vma_t));
    ch->vma_cnt=par->vma_cnt; ch->brk=par->brk;
    ch->stack_top=par->stack_top; ch->mmap_base=par->mmap_base;
    ch->aslr_seed=par->aslr_seed; ch->asid=vm_alloc_asid();
    ch->vsz=par->vsz;
    for (uint32_t gi=0; gi<PGD_SIZE; gi++) {
        if (!par->pgd[gi]) continue;
        if (par->pgd[gi] & PGD_HUGE_BIT) {
            ch->pgd[gi]=par->pgd[gi]; ch->huge_pte[gi]=par->huge_pte[gi];
            uint32_t fi=PTE_FIDX(par->huge_pte[gi]);
            if (fi) { frame_t *f=fidx_get(fi); if (f) f->ref_count++; }
            continue;
        }
        uint8_t nt=pt_slab_alloc(); if (!nt) break;
        ch->pgd[gi]=nt;
        pt_page_t *src=&g_pt_slab[par->pgd[gi]-1u];
        pt_page_t *dst=&g_pt_slab[nt-1u];
        memcpy(dst, src, sizeof(pt_page_t));
        for (int pi=0; pi<(int)PT_SIZE; pi++) {
            pte_t *cp=&dst->e[pi], *pp=&src->e[pi];
            if (!*cp) continue;
            uint32_t fi=PTE_FIDX(*cp); if (!fi) continue;
            frame_t *f=fidx_get(fi); if (!f) continue;
            if (PTE_FLAGS(*cp) & PF_SHR) {
                f->ref_count++;
            } else {
                f->ref_count++;
                if (PTE_FLAGS(*cp) & PF_W) {
                    uint32_t bb=*cp & ~(uint32_t)(PF_W|0x7Fu);
                    uint8_t  npf=(uint8_t)(PTE_FLAGS(*cp)&~PF_W);
                    *cp=bb|PTE_COW|npf; *pp=bb|PTE_COW|npf;
                }
            }
        }
    }
    ch->rss=par->rss;
    tlb_flush_asid(ch->asid);
    tlb_flush_asid(par->asid);
    return ch;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §16  vm_destroy
 * ═══════════════════════════════════════════════════════════════════════════ */
static void vm_destroy(vm_space_t *vm) {
    if (!vm) return;
    for (uint32_t gi=0; gi<PGD_SIZE; gi++) {
        if (!vm->pgd[gi]) continue;
        if (vm->pgd[gi] & PGD_HUGE_BIT) {
            uint32_t fi=PTE_FIDX(vm->huge_pte[gi]);
            if (fi) fidx_release(fi);
            vm->pgd[gi]=0; continue;
        }
        pt_page_t *pt=&g_pt_slab[vm->pgd[gi]-1u];
        for (int pi=0; pi<(int)PT_SIZE; pi++) {
            pte_t pv=pt->e[pi]; if (!pv) continue;
            uint32_t fi=PTE_FIDX(pv); if (fi) fidx_release(fi);
        }
        pt_slab_free(vm->pgd[gi]); vm->pgd[gi]=0;
    }
    tlb_flush_asid(vm->asid);
    free(vm);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §17  brk / stack setup / /proc/maps
 * ═══════════════════════════════════════════════════════════════════════════ */
static uint32_t vm_brk(vm_space_t *vm, uint32_t new_brk, int pid) {
    (void)pid;
    if (!new_brk) return vm->brk;
    if (new_brk > vm->brk) {
        uint32_t a = vm_mmap(vm, vm->brk, new_brk-vm->brk,
                             PROT_READ|PROT_WRITE|PROT_NOEXEC,
                             MAP_PRIVATE|MAP_ANONYMOUS, -1, 0, 0);
        vma_t *v=vm_vma_find(vm,a);
        if (v) strncpy(v->label,"[heap]",15);
    }
    vm->brk=new_brk;
    return vm->brk;
}

/* cpu_brk — user-mode aware; heap VMA gets MM_USER when cpl==3 */
static uint32_t cpu_brk(cpu_ctx_t *cpu, uint32_t new_brk) {
    vm_space_t *vm=cpu->vm;
    if (!new_brk) return vm->brk;
    if (new_brk > vm->brk) {
        uint16_t extra = (cpu->cpl==3) ? MM_USER : 0u;
        uint32_t a = vm_mmap(vm, vm->brk, new_brk-vm->brk,
                             PROT_READ|PROT_WRITE|PROT_NOEXEC,
                             MAP_PRIVATE|MAP_ANONYMOUS, -1, 0, extra);
        vma_t *v=vm_vma_find(vm,a);
        if (v) strncpy(v->label,"[heap]",15);
    }
    vm->brk=new_brk;
    return vm->brk;
}

static void vm_setup_stack(vm_space_t *vm, int pid) {
    (void)pid;
    uint32_t guard = vm->stack_top - (STACK_PAGES+1u)*PAGE_SIZE;
    vm_add_vma(vm, guard, guard+PAGE_SIZE,
               MM_GUARD, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0, "[guard]");
    uint32_t ss = vm->stack_top - STACK_PAGES*PAGE_SIZE;
    vm_mmap(vm, ss, STACK_PAGES*PAGE_SIZE,
            PROT_READ|PROT_WRITE|PROT_NOEXEC,
            MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE, -1, 0, 0);
    vma_t *v=vm_vma_find(vm,ss);
    if (v) strncpy(v->label,"[stack]",15);
}

/* cpu_setup_stack — stack VMA gets MM_USER when cpl==3 */
static void cpu_setup_stack(cpu_ctx_t *cpu) {
    vm_space_t *vm = cpu->vm;
    uint32_t guard = vm->stack_top - (STACK_PAGES+1u)*PAGE_SIZE;
    vm_add_vma(vm, guard, guard+PAGE_SIZE,
               MM_GUARD, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0, "[guard]");
    uint32_t ss = vm->stack_top - STACK_PAGES*PAGE_SIZE;
    uint16_t extra = (cpu->cpl==3) ? MM_USER : 0u;
    vm_mmap(vm, ss, STACK_PAGES*PAGE_SIZE,
            PROT_READ|PROT_WRITE|PROT_NOEXEC,
            MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE, -1, 0, extra);
    vma_t *v = vm_vma_find(vm, ss);
    if (v) strncpy(v->label,"[stack]",15);
}

static void vm_print_maps(vm_space_t *vm, int pid) {
    printf("/proc/%d/maps:  RSS=%u pages  VSZ=%u pages\n", pid, vm->rss, vm->vsz);
    for (uint8_t i=0; i<vm->vma_cnt; i++) {
        vma_t *v=&vm->vmas[i];
        char r=(v->mm_flags&MM_READ)?'r':'-', w=(v->mm_flags&MM_WRITE)?'w':'-';
        char x=(v->mm_flags&MM_EXEC)?'x':'-',  p=(v->mm_flags&MM_SHARED)?'s':'p';
        char u=(v->mm_flags&MM_USER)?'U':'K';   /* v7: show U/K privilege  */
        printf("  %08x-%08x %c%c%c%c %c %08x 00:00 0 %s\n",
               v->start, v->end, r,w,x,p, u, v->offset, v->label);
    }
}

/* ── §18  RAM usage report ──────────────────────────────────────────────── */
static void mmu_print_stats(void) {
    uint32_t pt_used=(uint32_t)__builtin_popcountll(g_pt_bmap), fr_used=0;
    for (uint32_t i=0;i<MAX_FRAMES;i++)
        if (g_frames[i].ref_count>0) fr_used++;
    printf("[MMU stats] PT slab: %u/%u tables (%u KB)  "
           "Frames: %u/%u (%u KB phys)\n",
           pt_used, MAX_PT_TABLES, pt_used*4u,
           fr_used, MAX_FRAMES,   fr_used*4u);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §19  Self-test  (demonstrates all three v7 features)
 * ═══════════════════════════════════════════════════════════════════════════ */
#ifdef MMU_TEST

/* Custom fault handler: logs faults, resolves none (lets default kill) */
static void test_fault_handler(cpu_ctx_t *cpu, fault_record_t *rec) {
    static const char *names[] = {
        "NONE","UNMAPPED","GUARD","STACK_OVF",
        "PERM_R","PERM_W","PERM_X","PERM_PRIV","OOM"
    };
    const char *n = (rec->type < (fault_type_t)9) ? names[rec->type] : "?";
    printf("  [TRAP #PF] %-12s  cpl=%d  addr=0x%08X  %s\n",
           n, rec->cpl, rec->addr, rec->is_write?"WRITE":"READ");
    cpu->killed = true;
}

int main(void) {
    puts("══ MMU v7 self-test ══════════════════════════════════════════════");

    /* ── Test 1: Automatic translation ───────────────────────────────────── *
     *  cpu_store_u32 / cpu_load_u32 do full VA→PA automatically.
     *  No explicit vm_fault() call needed anywhere.                          */
    puts("\n[1] Automatic translation — cpu_store_u32 / cpu_load_u32");
    vm_space_t *vm = vm_create();
    cpu_ctx_t   cpu; cpu_init(&cpu, vm, 1, 3);  /* user-mode process        */
    cpu.fault_handler = test_fault_handler;

    cpu_setup_stack(&cpu);
    cpu_brk(&cpu, vm->brk + PAGE_SIZE*4);        /* 4-page heap              */

    uint32_t heap = vm->brk - PAGE_SIZE*4;
    cpu_store_u32(&cpu, heap,   0xDEADBEEFu);
    cpu_store_u32(&cpu, heap+4, 0xCAFEBABEu);
    uint32_t a = cpu_load_u32(&cpu, heap);
    uint32_t b = cpu_load_u32(&cpu, heap+4);
    printf("  heap[0]=0x%08X  heap[4]=0x%08X  %s\n",
           a, b, (a==0xDEADBEEFu && b==0xCAFEBABEu)?"OK":"FAIL");

    /* ── Test 2: Fault trap — unmapped address ───────────────────────────── */
    puts("\n[2] Fault trap — UNMAPPED access");
    cpu.killed = false;
    uint32_t bad = cpu_load_u32(&cpu, 0x12340000u);
    (void)bad;
    printf("  cpu.killed=%d  fault=%s  addr=0x%08X  %s\n",
           cpu.killed, cpu.last_fault.type==FAULT_UNMAPPED?"UNMAPPED":"?",
           cpu.last_fault.addr,
           (cpu.killed && cpu.last_fault.type==FAULT_UNMAPPED)?"OK":"FAIL");

    /* ── Test 3: Fault trap — write to read-only mapping ─────────────────── */
    puts("\n[3] Fault trap — PERM_W (write to RO page)");
    cpu.killed = false;
    uint32_t ro_addr = cpu_mmap(&cpu, 0, PAGE_SIZE,
                                PROT_READ, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    cpu_load_u8(&cpu, ro_addr);   /* trigger demand fault (read = ok)  */
    cpu.killed = false;
    cpu_store_u8(&cpu, ro_addr, 0xAA);
    printf("  cpu.killed=%d  fault=%s  %s\n",
           cpu.killed,
           cpu.last_fault.type==FAULT_PERM_W?"PERM_W":"?",
           (cpu.killed && cpu.last_fault.type==FAULT_PERM_W)?"OK":"FAIL");

    /* ── Test 4: Privilege enforcement — user accessing kernel VMA ────────── *
     *  Allocate a kernel VMA (no MM_USER).  User CPU (CPL=3) must be denied. */
    puts("\n[4] Privilege enforcement — user CPL=3 accessing kernel VMA");
    /* Create kernel VMA by bypassing cpu_mmap (no MM_USER added)            */
    uint32_t kaddr = vm_mmap(vm, 0, PAGE_SIZE,
                             PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS,
                             -1, 0, /*extra_mmf=*/0);  /* no MM_USER          */
    /* Prime it from kernel context */
    {   cpu_ctx_t kcpu; cpu_init(&kcpu, vm, 1, 0);  /* CPL=0 kernel         */
        cpu_store_u32(&kcpu, kaddr, 0xC0DEC0DEu);    /* kernel writes OK     */
        uint32_t kv = cpu_load_u32(&kcpu, kaddr);
        printf("  kernel write+read 0x%08X  %s\n", kv,
               kv==0xC0DEC0DEu?"OK":"FAIL"); }
    /* Now try user access — must be denied */
    cpu.killed = false;
    uint32_t kv = cpu_load_u32(&cpu, kaddr);  /* CPL=3 → FAULT_PERM_PRIV   */
    (void)kv;
    printf("  user read blocked: cpu.killed=%d  fault=%s  %s\n",
           cpu.killed,
           cpu.last_fault.type==FAULT_PERM_PRIV?"PERM_PRIV":"?",
           (cpu.killed && cpu.last_fault.type==FAULT_PERM_PRIV)?"OK":"FAIL");

    /* ── Test 5: NX fault ────────────────────────────────────────────────── */
    puts("\n[5] Fault trap — PERM_X (execute NX page)");
    cpu.killed = false;
    cpu_mmap(&cpu, 0, PAGE_SIZE, PROT_READ|PROT_WRITE|PROT_NOEXEC,
             MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    /* vm_fault exec path */
    uint32_t nx_addr = vm->mmap_base - PAGE_SIZE;
    /* trigger via vm_fault directly to test exec flag */
    vm_fault(&cpu, nx_addr-PAGE_SIZE*2, 0, /*exec=*/1);
    printf("  fault=%s  %s\n",
           cpu.last_fault.type!=FAULT_NONE ? g_fault_names[cpu.last_fault.type] : "NONE",
           cpu.killed?"blocked":"miss");

    /* ── Summary ─────────────────────────────────────────────────────────── */
    puts("");
    vm_print_maps(vm, 1);
    mmu_print_stats();
    vm_destroy(vm);
    puts("══ done ══════════════════════════════════════════════════════════");
    return 0;
}
#endif /* MMU_TEST */
