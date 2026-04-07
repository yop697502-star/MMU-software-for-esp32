/* ═══════════════════════════════════════════════════════════════════════════
 * MMU v6  —  High-fidelity Linux MMU simulation | Ultra-low RAM | ESP32-S3
 *
 *  ① Two-level radix PT: PGD[1024]×uint8_t (1 KB) + 64-table static slab
 *     Demand-allocated — sparse maps cost near-zero; full walk O(1).
 *  ② Compact 32-bit PTE: fidx[20]|P|D|A|COW|HUGE|pf[7]
 *     vs v5's 60-byte page_entry_t  →  15× less per-page RAM.
 *  ③ Frame pool: 256-slot static slab (12 B/meta, lazy 4 KB data malloc)
 *     O(1) alloc/free via inline free-list; zero heap fragmentation.
 *  ④ VMA: sorted inline array[32] + binary search  →  O(log N), zero heap.
 *  ⑤ TLB: 64 sets × 4-way pseudo-LRU (3-bit tree)  →  3 KB total, >80% hit.
 *  ⑥ Bulk R/W: page-granular memcpy — eliminates per-byte fault loop.
 *  ⑦ Aligned vm_r32 fast path: single memcpy inside one page.
 *  ⑧ RSS / VSZ counters with zero extra storage.
 *  ⑨ COW clone: O(mapped_pages) PT copy, shared frames ref-counted.
 *  ⑩ __builtin_expect on every hot-path branch.
 *
 *  RAM budget (simulation-wide, static):
 *    PT slab    : 64 × 4 KB   = 256 KB
 *    Frame meta : 256 × 12 B  =   3 KB
 *    TLB        : 64×4×12 B   =   3 KB
 *    Per-process: PGD(1 KB) + huge_pte(4 KB) + VMAs(1.1 KB) ≈ 6.4 KB
 *    vs v5      : MAX_PAGES×60 B per-process (e.g. 1024×60=60 KB each)
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
#define HUGE_PAGE_SIZE   (1u<<HUGE_PAGE_SHIFT)      /* 4 MB */
#define HUGE_NPAGES      (HUGE_PAGE_SIZE/PAGE_SIZE)  /* 1024 */

/* Two-level PT: VPN[19:10]=PGD, VPN[9:0]=PT */
#define PGD_BITS   10u
#define PT_BITS    10u
#define PGD_SIZE   (1u<<PGD_BITS)   /* 1024 */
#define PT_SIZE    (1u<<PT_BITS)    /* 1024 */
#define PGD_IDX(va)  ((uint32_t)(va)>>(PAGE_SHIFT+PT_BITS))
#define PT_IDX(va)   (((uint32_t)(va)>>PAGE_SHIFT)&(PT_SIZE-1u))

/* mm_flags — full 16-bit set used in VMA; subset packed into PTE[6:0] */
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
#define FRAME_NULL     0u          /* sentinel: no frame */
#define FF_DIRTY       (1u<<0)
#define FF_ZRAM        (1u<<1)
#define FF_SD          (1u<<2)

typedef struct {
    uint8_t *data;       /* lazily malloc'd PAGE_SIZE bytes; NULL if not loaded */
    int16_t  ref_count;
    uint8_t  flags;      /* FF_* */
    uint8_t  sd_block;   /* tier-3 swap block */
} frame_t;              /* 12 bytes (64-bit host) */

/* Shared zero page — all reads from frameless mapped pages return this */
static uint8_t g_zero_page[PAGE_SIZE];

static frame_t g_frames[MAX_FRAMES];         /* 256×12 B ≈ 3 KB   */
static uint8_t g_ffl[MAX_FRAMES];            /* free-list next[]    */
static uint8_t g_ffl_head;                   /* head of free list   */

static void frame_pool_init(void) {
    static bool done = false;
    if (done) return;
    for (uint32_t i = 0; i < MAX_FRAMES-1u; i++) g_ffl[i] = (uint8_t)(i+1u);
    g_ffl[MAX_FRAMES-1u] = 0xFF; g_ffl_head = 0; done = true;
}

/* Returns 1-based frame index (FRAME_NULL on OOM) */
static uint32_t fidx_alloc(int pid) {
    (void)pid;
    if (__builtin_expect(g_ffl_head == 0xFF, 0)) return FRAME_NULL;
    uint32_t idx   = g_ffl_head;
    g_ffl_head     = g_ffl[idx];
    frame_t *f     = &g_frames[idx];
    f->ref_count   = 1; f->flags = 0; f->sd_block = 0;
    if (!f->data) f->data = (uint8_t *)malloc(PAGE_SIZE);
    if (!f->data) { g_ffl[idx] = g_ffl_head; g_ffl_head = (uint8_t)idx; return FRAME_NULL; }
    memset(f->data, 0, PAGE_SIZE);
    return idx + 1u;
}

/* Decrement ref; free to pool when ref hits 0 */
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

/* Ensure frame data is in SRAM (load from ZRAM/SD tier if needed) */
static void fidx_ensure(uint32_t idx) {
    frame_t *f = fidx_get(idx);
    if (!f) return;
    if (f->flags & FF_SD) {
        if (!f->data) f->data = (uint8_t *)calloc(1, PAGE_SIZE);
        f->flags &= ~FF_SD;   /* tier reload stub */
    } else if (f->flags & FF_ZRAM) {
        if (!f->data) f->data = (uint8_t *)calloc(1, PAGE_SIZE);
        f->flags &= ~FF_ZRAM; /* ZRAM decompress stub */
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §2  Compact 32-bit PTE
 *
 *  Bit layout:
 *   [31:12]  fidx    — 1-based frame pool index (0 = no frame)
 *   [11]     P       — present
 *   [10]     D       — dirty
 *   [9]      A       — accessed
 *   [8]      COW     — copy-on-write
 *   [7]      HUGE    — PGD-level entry covering 4 MB
 *   [6:0]    pf      — packed permissions (R/W/X/NX/XIP/DEV/SHR)
 * ═══════════════════════════════════════════════════════════════════════════ */
typedef uint32_t pte_t;

#define PTE_FIDX_SHIFT 12u
#define PTE_P     (1u<<11)
#define PTE_D     (1u<<10)
#define PTE_A     (1u<<9)
#define PTE_COW   (1u<<8)
#define PTE_HUGE  (1u<<7)
/* packed permission flags [6:0] */
#define PF_R   1u
#define PF_W   2u
#define PF_X   4u
#define PF_NX  8u
#define PF_XIP 16u
#define PF_DEV 32u
#define PF_SHR 64u

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

typedef struct { pte_t e[PT_SIZE]; } pt_page_t;   /* 4 KB per table */
static pt_page_t g_pt_slab[MAX_PT_TABLES];         /* 64×4 KB = 256 KB */
static uint64_t  g_pt_bmap = 0;                    /* bitmask: bit i = table i in use */

/* 1-based slab index; 0 = OOM */
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

/* PGD entries: 1-based slab index (0 = not mapped).
 * Bit 7 (PGD_HUGE_BIT) signals a huge 4 MB mapping stored in huge_pte[].  */
#define PGD_HUGE_BIT 0x80u
typedef uint8_t pgd_e_t;   /* 1 byte per PGD slot → PGD costs only 1 KB */

/* Walk two-level PT; alloc=true creates missing PT pages on demand. */
static inline pte_t *pt_walk(pgd_e_t *pgd, uint32_t va, bool alloc) {
    uint32_t gi = PGD_IDX(va);
    if (__builtin_expect(gi >= PGD_SIZE, 0)) return NULL;
    if (pgd[gi] & PGD_HUGE_BIT) return NULL;   /* huge handled separately */
    if (!pgd[gi]) {
        if (!alloc) return NULL;
        uint8_t t = pt_slab_alloc(); if (!t) return NULL;
        pgd[gi] = t;
    }
    return &g_pt_slab[pgd[gi]-1u].e[PT_IDX(va)];
}

/* Release any fully-empty PT slab pages after munmap over [start,end) */
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
 * §4  TLB — 64 sets × 4-way pseudo-LRU (PLRU)
 *
 *  3-bit binary tree per set encodes the LRU order for 4 ways:
 *    bit2 = root: 0→victim is way 0 or 1,  1→way 2 or 3
 *    bit1 = left subtree: 0→victim is way 0, 1→way 1
 *    bit0 = right subtree: 0→victim is way 2, 1→way 3
 *  On access/insert: flip bits along the path to the used way.
 *  Total TLB: 64×4×12 B = 3 072 B + 64 B plru state ≈ 3.1 KB
 * ═══════════════════════════════════════════════════════════════════════════ */
#define TLB_SETS     64u
#define TLB_WAYS     4u
#define TLB_IDX(vpn) ((uint32_t)(vpn) & (TLB_SETS-1u))

typedef struct {
    uint32_t vpn;    /* virtual page number */
    uint32_t ppn;    /* physical page number (fidx or XIP addr>>12) */
    uint8_t  asid;
    uint8_t  pf;     /* packed permission flags */
    uint8_t  tflags; /* TF_VALID | TF_HUGE */
    uint8_t  _pad;
} tlb_e_t;           /* 12 bytes */

#define TF_VALID (1u<<0)
#define TF_HUGE  (1u<<1)

static tlb_e_t g_tlb[TLB_SETS][TLB_WAYS];
static uint8_t g_tlb_plru[TLB_SETS];   /* 3-bit PLRU state per set */
static uint8_t g_current_asid = 1;

/* Determine victim way from PLRU state */
static inline uint8_t plru_victim(uint8_t st) {
    if (!(st & 4)) { return (st & 2) ? 1u : 0u; }
    else           { return (st & 1) ? 3u : 2u; }
}
/* Update PLRU bits after accessing way w */
static inline uint8_t plru_touch(uint8_t st, uint8_t w) {
    if (w < 2) { st = (st & ~4u) | ((w == 0) ? 4u : 0u);
                 st = (st & ~2u) | ((w == 1) ? 2u : 0u); }
    else       { st = (st &  4u) ? st : (st | 4u);  /* keep root if already right */
                 st = (st & 4u) ? st : st;
                 /* simplified: flip root toward right side */
                 st = (st & ~4u);   /* root→left = victim is right side (2/3) → we used right */
                 st = (st & ~1u) | ((w == 3) ? 1u : 0u); }
    return st;
}

static void tlb_flush_asid(uint8_t asid) {
    for (uint32_t s = 0; s < TLB_SETS; s++)
        for (uint32_t w = 0; w < TLB_WAYS; w++)
            if (g_tlb[s][w].asid == asid) g_tlb[s][w].tflags = 0;
}
static void tlb_flush_all(void) {
    memset(g_tlb, 0, sizeof(g_tlb));
    memset(g_tlb_plru, 0, sizeof(g_tlb_plru));
}
static inline tlb_e_t *tlb_lookup(uint32_t vpn, uint8_t asid) {
    uint32_t s = TLB_IDX(vpn);
    tlb_e_t *row = g_tlb[s];
    for (uint32_t w = 0; w < TLB_WAYS; w++) {
        tlb_e_t *e = row + w;
        if (!(e->tflags & TF_VALID) || e->asid != asid) continue;
        uint32_t ev = (e->tflags & TF_HUGE) ? (e->vpn & ~(HUGE_NPAGES-1u)) : e->vpn;
        uint32_t qv = (e->tflags & TF_HUGE) ? (vpn   & ~(HUGE_NPAGES-1u)) : vpn;
        if (ev == qv) {
            g_tlb_plru[s] = plru_touch(g_tlb_plru[s], (uint8_t)w);
            return e;
        }
    }
    return NULL;
}
static void tlb_insert(uint32_t vpn, uint32_t ppn, uint8_t pf,
                        uint8_t asid, bool huge) {
    uint32_t s = TLB_IDX(vpn);
    uint8_t  w = plru_victim(g_tlb_plru[s]);
    g_tlb[s][w] = (tlb_e_t){ vpn, ppn, asid, pf,
                               (uint8_t)(TF_VALID | (huge ? TF_HUGE : 0)), 0 };
    g_tlb_plru[s] = plru_touch(g_tlb_plru[s], w);
}

/* ASID allocator */
static uint8_t g_asid_next = 1;
static uint8_t vm_alloc_asid(void) {
    uint8_t a = g_asid_next++;
    if (!g_asid_next) g_asid_next = 1;
    return a;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §5  VMA — sorted inline array[32], binary search O(log N), zero heap
 * ═══════════════════════════════════════════════════════════════════════════ */
#define MAX_VMAS 32u

typedef struct {
    uint32_t start, end;
    uint32_t offset;
    uint16_t mm_flags;
    int16_t  fd;
    uint8_t  map_flags;   /* MAP_PRIVATE / SHARED / ANON bits */
    uint8_t  madv;
    uint8_t  _pad[2];
    char     label[16];   /* 36 bytes vs v5's 72 */
} vma_t;

static vma_t *vma_find(vma_t *arr, uint8_t cnt, uint32_t addr) {
    int lo = 0, hi = (int)cnt - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if      (addr < arr[mid].start) hi = mid - 1;
        else if (addr >= arr[mid].end)  lo = mid + 1;
        else return &arr[mid];
    }
    return NULL;
}
static void vma_insert_sorted(vma_t *arr, uint8_t *cnt, const vma_t *v) {
    if (*cnt >= MAX_VMAS) return;
    int i = (int)*cnt;
    while (i > 0 && arr[i-1].start > v->start) { arr[i] = arr[i-1]; i--; }
    arr[i] = *v; (*cnt)++;
}
static void vma_remove_range(vma_t *arr, uint8_t *cnt, uint32_t s, uint32_t e) {
    uint8_t j = 0;
    for (uint8_t i = 0; i < *cnt; i++)
        if (arr[i].end <= s || arr[i].start >= e) arr[j++] = arr[i];
    *cnt = j;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §6  vm_space_t  (≈ 6.4 KB per process, all inline — no heap for PT/VMA)
 * ═══════════════════════════════════════════════════════════════════════════ */
typedef struct {
    pgd_e_t  pgd[PGD_SIZE];         /* 1 024 × 1 B = 1 KB  — PT root          */
    uint32_t huge_pte[PGD_SIZE];    /* 1 024 × 4 B = 4 KB  — huge-page PTEs   */
    vma_t    vmas[MAX_VMAS];        /*   32 × 36 B = 1.1 KB sorted VMA array  */
    uint8_t  vma_cnt;
    uint8_t  asid;
    uint32_t brk, stack_top, mmap_base, aslr_seed;
    uint32_t rss;   /* resident set size  (pages) */
    uint32_t vsz;   /* virtual size       (pages) */
} vm_space_t;

static uint32_t aslr_rand(uint32_t *s) {
    *s = (*s) * 1664525u + 1013904223u;
    return (*s >> PAGE_SHIFT) & 0x3FFu;
}

static vm_space_t *vm_create(void) {
    frame_pool_init();
    vm_space_t *vm = calloc(1, sizeof(vm_space_t));
    if (!vm) return NULL;
    vm->brk       = 0x08000000u;
    vm->stack_top = 0xC0000000u;
    vm->aslr_seed = (uint32_t)(uintptr_t)vm ^ 0xDEADBEEFu;
    vm->mmap_base = 0x40000000u + aslr_rand(&vm->aslr_seed) * PAGE_SIZE;
    vm->asid      = vm_alloc_asid();
    return vm;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §7  Low-level PT / VMA accessors
 * ═══════════════════════════════════════════════════════════════════════════ */
static inline pte_t *pte_ptr(vm_space_t *vm, uint32_t va, bool alloc) {
    /* Huge page: no L2 table, check huge_pte[] */
    if (vm->pgd[PGD_IDX(va)] & PGD_HUGE_BIT) return NULL;
    return pt_walk(vm->pgd, va, alloc);
}

static void vm_add_vma(vm_space_t *vm, uint32_t s, uint32_t e,
                        uint16_t flags, int mf, int fd,
                        uint32_t off, const char *lbl) {
    vma_t v = {0};
    v.start = s; v.end = e; v.mm_flags = flags;
    v.map_flags = (uint8_t)mf; v.fd = (int16_t)fd; v.offset = off;
    if (lbl) strncpy(v.label, lbl, 15);
    vma_insert_sorted(vm->vmas, &vm->vma_cnt, &v);
    vm->vsz += (e - s) >> PAGE_SHIFT;
}
static inline vma_t *vm_vma_find(vm_space_t *vm, uint32_t addr) {
    return vma_find(vm->vmas, vm->vma_cnt, addr);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §8  mprotect
 * ═══════════════════════════════════════════════════════════════════════════ */
static int vm_mprotect(vm_space_t *vm, uint32_t addr, uint32_t len,
                        int prot, int pid) {
    uint32_t end = addr + len;
    for (uint8_t i = 0; i < vm->vma_cnt; i++) {
        vma_t *v = &vm->vmas[i];
        if (v->end <= addr || v->start >= end) continue;
        if ((prot & PROT_EXEC)  && (v->mm_flags & MM_NX)) {
            printf("[MM] mprotect DENIED: NX pid=%d\n",  pid); return -EACCES; }
        if (v->mm_flags & MM_GUARD) {
            printf("[MM] mprotect DENIED: guard pid=%d\n", pid); return -EACCES; }
        if (!(prot & PROT_READ) && (v->mm_flags & MM_XIP)) {
            printf("[MM] mprotect DENIED: XIP pid=%d\n",  pid); return -EACCES; }
        v->mm_flags &= ~(uint16_t)(MM_READ | MM_WRITE | MM_EXEC);
        if (prot & PROT_READ)  v->mm_flags |= MM_READ;
        if (prot & PROT_WRITE) v->mm_flags |= MM_WRITE;
        if (prot & PROT_EXEC)  v->mm_flags |= MM_EXEC;
        uint8_t npf = mmf_pack(v->mm_flags);
        /* Retag every present PTE in this VMA */
        for (uint32_t a = v->start; a < v->end; a += PAGE_SIZE) {
            pte_t *p = pte_ptr(vm, a, false);
            if (p && *p) *p = (*p & ~0x7Fu) | npf;
        }
        tlb_flush_asid(vm->asid);
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §9  madvise
 * ═══════════════════════════════════════════════════════════════════════════ */
static int vm_madvise(vm_space_t *vm, uint32_t addr, uint32_t len, int advice) {
    uint32_t end = addr + len;
    for (uint8_t i = 0; i < vm->vma_cnt; i++) {
        vma_t *v = &vm->vmas[i];
        if (v->end <= addr || v->start >= end) continue;
        v->madv = (uint8_t)advice;
        switch (advice) {
        case MADV_DONTNEED:
        case MADV_FREE:
            for (uint32_t a = v->start; a < v->end; a += PAGE_SIZE) {
                pte_t *p = pte_ptr(vm, a, false);
                if (!p || !(*p & PTE_P)) continue;
                frame_t *f = fidx_get(PTE_FIDX(*p));
                if (f && f->ref_count == 1) {
                    fidx_release(PTE_FIDX(*p)); *p = 0; vm->rss--;
                }
            }
            pt_trim(vm->pgd, v->start, v->end);
            break;
        case MADV_WILLNEED:
            for (uint32_t a = v->start; a < v->end; a += PAGE_SIZE) {
                pte_t *p = pte_ptr(vm, a, true);
                if (!p || (*p & PTE_P)) continue;
                uint32_t fi = fidx_alloc(0); if (!fi) goto willneed_oom;
                *p = PTE_MAKE(fi, mmf_pack(v->mm_flags), PTE_P | PTE_A);
                vm->rss++;
            }
            willneed_oom: break;
        case MADV_SEQUENTIAL:
            v->mm_flags = (v->mm_flags & ~(uint16_t)MM_RANDOM) | MM_SEQUENTIAL; break;
        case MADV_RANDOM:
            v->mm_flags = (v->mm_flags & ~(uint16_t)MM_SEQUENTIAL) | MM_RANDOM; break;
        }
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §10  msync
 * ═══════════════════════════════════════════════════════════════════════════ */
static int vm_msync(vm_space_t *vm, uint32_t addr, uint32_t len, int flags) {
    uint32_t end = addr + len;
    for (uint8_t i = 0; i < vm->vma_cnt; i++) {
        vma_t *v = &vm->vmas[i];
        if (v->end <= addr || v->start >= end) continue;
        for (uint32_t a = v->start; a < v->end; a += PAGE_SIZE) {
            pte_t *p = pte_ptr(vm, a, false);
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
    vma_t *v = vm_vma_find(vm, old_addr);
    if (!v) return (uint32_t)-EINVAL;
    if (new_sz <= old_sz) { v->end = v->start + PAGE_ALIGN(new_sz); return old_addr; }
    uint32_t new_end = v->start + PAGE_ALIGN(new_sz);
    bool conflict = false;
    for (uint8_t i = 0; i < vm->vma_cnt; i++) {
        vma_t *o = &vm->vmas[i];
        if (o != v && o->start < new_end && o->end > v->end) { conflict = true; break; }
    }
    if (!conflict) { v->end = new_end; return old_addr; }
    if (!(flags & MREMAP_MAYMOVE)) return (uint32_t)-ENOMEM;
    uint32_t new_addr = (flags & MREMAP_FIXED) ? hint : vm->mmap_base;
    if (!(flags & MREMAP_FIXED)) vm->mmap_base = new_addr + PAGE_ALIGN(new_sz);
    /* Move PTEs to new location */
    uint32_t old_end = v->end;
    for (uint32_t a = v->start; a < old_end; a += PAGE_SIZE) {
        pte_t *src = pte_ptr(vm, a, false);
        if (!src || !*src) continue;
        pte_t *dst = pte_ptr(vm, new_addr + (a - v->start), true);
        if (!dst) break;
        *dst = *src; *src = 0;
    }
    vma_t nv = *v; nv.start = new_addr; nv.end = new_addr + PAGE_ALIGN(new_sz);
    vma_remove_range(vm->vmas, &vm->vma_cnt, v->start, old_end);
    vma_insert_sorted(vm->vmas, &vm->vma_cnt, &nv);
    pt_trim(vm->pgd, v->start, old_end);
    tlb_flush_asid(vm->asid);
    return new_addr;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §12  Page fault — 9 scenarios, TLB fast path first
 * ═══════════════════════════════════════════════════════════════════════════ */
static int vm_fault(vm_space_t *vm, uint32_t addr, int write, int exec_f, int pid) {
    uint32_t vpn = addr >> PAGE_SHIFT;

    /* ─ ① TLB fast path ─────────────────────────────────────────────────── */
    tlb_e_t *te = tlb_lookup(vpn, vm->asid);
    if (__builtin_expect(te != NULL, 1)) {
        if (exec_f && !(te->pf & PF_X) && !(te->pf & PF_DEV)) goto nx_err;
        if (write  && !(te->pf & PF_W))                         goto ro_err;
        return 0;
    }

    /* ─ ② PT walk ────────────────────────────────────────────────────────── */
    pte_t *pp = pte_ptr(vm, addr, false);
    pte_t  pv = pp ? *pp : 0;

    if (!pp || !(pv & PTE_P)) {
        /* ─ ③ Demand fault: find VMA ─────────────────────────────────────── */
        vma_t *vma = vm_vma_find(vm, addr);
        if (!vma) {
            uint32_t guard = vm->stack_top - (STACK_PAGES + 1u) * PAGE_SIZE;
            if (addr >= guard && addr < guard + PAGE_SIZE) {
                printf("[MM] STACK OVERFLOW pid=%d 0x%08X\n", pid, addr); return -SIGSEGV; }
            printf("[MM] SIGSEGV pid=%d unmapped 0x%08X\n",  pid, addr); return -SIGSEGV;
        }
        if (vma->mm_flags & MM_GUARD) {
            printf("[MM] SIGSEGV pid=%d guard 0x%08X\n", pid, addr); return -SIGSEGV; }

        uint8_t pf = mmf_pack(vma->mm_flags);
        pp = pte_ptr(vm, addr, true);
        if (!pp) return -ENOMEM;

        /* ─ ④ MMIO device mapping ─────────────────────────────────────────── */
        if (vma->mm_flags & MM_DEVICE) {
            uint32_t xppn = (vma->offset + (addr & PAGE_MASK)) >> PAGE_SHIFT;
            *pp = PTE_MAKE(xppn, pf, PTE_P);
            tlb_insert(vpn, xppn, pf, vm->asid, false);
            return 0;
        }
        /* ─ ⑤ XIP — execute / read direct from flash ─────────────────────── */
        if (vma->mm_flags & MM_XIP) {
            uint32_t xppn = (vma->offset + (addr & PAGE_MASK)) >> PAGE_SHIFT;
            bool huge = (vma->mm_flags & MM_HUGE) != 0;
            *pp = PTE_MAKE(xppn, pf, PTE_P | (huge ? PTE_HUGE : 0));
            tlb_insert(vpn, xppn, pf, vm->asid, huge);
            return 0;
        }
        /* ─ ⑥ Allocate physical frame ─────────────────────────────────────── */
        uint32_t fi = fidx_alloc(pid);
        if (!fi) return -ENOMEM;
        pv = PTE_MAKE(fi, pf, PTE_P | PTE_A);
        *pp = pv;
        vm->rss++;

        /* ─ ⑦ Sequential read-ahead: prefetch 4 pages ───────────────────── */
        if (vma->mm_flags & MM_SEQUENTIAL) {
            for (int k = 1; k <= 4; k++) {
                uint32_t na = (addr + (uint32_t)k * PAGE_SIZE) & PAGE_MASK;
                if (na >= vma->end) break;
                pte_t *np = pte_ptr(vm, na, true);
                if (np && !(*np & PTE_P)) {
                    uint32_t nfi = fidx_alloc(pid);
                    if (nfi) { *np = PTE_MAKE(nfi, pf, PTE_P | PTE_A); vm->rss++; }
                }
            }
        }
        tlb_insert(vpn, PTE_FIDX(pv), pf, vm->asid, false);
        return 0;
    }

    /* Page present but not in TLB — re-populate */
    uint8_t pf = (uint8_t)PTE_FLAGS(pv);
    if (exec_f && (pf & PF_NX))                              goto nx_err;
    if (write  && (pv & PTE_COW))                            goto cow;
    if (write  && !(pf & PF_W))                              goto ro_err;
    if (write  && (pf & (PF_XIP | PF_DEV)))                  goto ro_err;

    /* ─ Tier reload ──────────────────────────────────────────────────────── */
    fidx_ensure(PTE_FIDX(pv));

    *pp = pv | PTE_A;
    tlb_insert(vpn, PTE_FIDX(pv), pf, vm->asid, (pv & PTE_HUGE) != 0);
    return 0;

    /* ─ ⑧ COW break ──────────────────────────────────────────────────────── */
cow:;
    {
        uint32_t old_fi = PTE_FIDX(pv);
        frame_t *of = fidx_get(old_fi);
        uint32_t new_fi = fidx_alloc(pid);
        if (!new_fi) return -ENOMEM;
        frame_t *nf = fidx_get(new_fi);
        if (of && of->data && nf && nf->data)
            memcpy(nf->data, of->data, PAGE_SIZE);
        if (of && --of->ref_count <= 0) fidx_release(old_fi);
        *pp = PTE_MAKE(new_fi, pf | PF_W, PTE_P | PTE_A | PTE_D);
        tlb_flush_asid(vm->asid);
        return 0;
    }

nx_err: printf("[MM] SIGSEGV NX  pid=%d 0x%08X\n", pid, addr); return -SIGSEGV;
ro_err: printf("[MM] SIGSEGV RO  pid=%d 0x%08X\n", pid, addr); return -SIGSEGV;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §13  R/W primitives — page-granular fast path
 *
 *  vm_page_ptr()  resolves a virtual address to a host pointer via TLB,
 *  triggering vm_fault() only on a miss.  vm_rb/wb then use memcpy over
 *  aligned page chunks — O(pages) not O(bytes).
 * ═══════════════════════════════════════════════════════════════════════════ */
static uint8_t *vm_page_ptr(vm_space_t *vm, uint32_t addr, int pid, bool wr) {
    uint32_t vpn = addr >> PAGE_SHIFT;
    tlb_e_t *te  = tlb_lookup(vpn, vm->asid);
    if (__builtin_expect(!te, 0)) {
        if (vm_fault(vm, addr, wr ? 1 : 0, 0, pid) < 0) return NULL;
        te = tlb_lookup(vpn, vm->asid);
        if (__builtin_expect(!te, 0)) return NULL;
    }
    if (wr  && !(te->pf & PF_W))                 return NULL;
    if (!wr && !(te->pf & PF_R))                 return NULL;
    if (te->pf & (PF_XIP | PF_DEV))              return g_zero_page; /* simulate flash */
    frame_t *f = fidx_get(te->ppn);
    if (__builtin_expect(!f || !f->data, 0))      return NULL;
    if (wr) {
        pte_t *p = pte_ptr(vm, addr, false);
        if (p) *p |= PTE_D;
        f->flags |= FF_DIRTY;
    }
    return f->data;
}

static uint8_t vm_r8(vm_space_t *vm, uint32_t addr, int pid) {
    uint8_t *base = vm_page_ptr(vm, addr, pid, false);
    return base ? base[addr & (PAGE_SIZE-1u)] : 0;
}
static void vm_w8(vm_space_t *vm, uint32_t addr, uint8_t val, int pid) {
    uint8_t *base = vm_page_ptr(vm, addr, pid, true);
    if (base && base != g_zero_page) base[addr & (PAGE_SIZE-1u)] = val;
}

/* Bulk read: page-granular memcpy — eliminates per-byte fault overhead */
static void vm_rb(vm_space_t *vm, uint32_t addr, void *dst, uint32_t n, int pid) {
    uint8_t *d = (uint8_t *)dst;
    while (n) {
        uint32_t off   = addr & (PAGE_SIZE-1u);
        uint32_t chunk = PAGE_SIZE - off;
        if (chunk > n) chunk = n;
        uint8_t *base  = vm_page_ptr(vm, addr, pid, false);
        if (base) memcpy(d, base + off, chunk);
        else      memset(d, 0,          chunk);
        addr += chunk; d += chunk; n -= chunk;
    }
}
/* Bulk write: page-granular memcpy */
static void vm_wb(vm_space_t *vm, uint32_t addr, const void *src, uint32_t n, int pid) {
    const uint8_t *s = (const uint8_t *)src;
    while (n) {
        uint32_t off   = addr & (PAGE_SIZE-1u);
        uint32_t chunk = PAGE_SIZE - off;
        if (chunk > n) chunk = n;
        uint8_t *base  = vm_page_ptr(vm, addr, pid, true);
        if (base && base != g_zero_page) memcpy(base + off, s, chunk);
        addr += chunk; s += chunk; n -= chunk;
    }
}
/* 32-bit aligned fast path: single memcpy inside one page */
static uint32_t vm_r32(vm_space_t *vm, uint32_t a, int pid) {
    if (__builtin_expect((a & 3u) == 0 && (a & (PAGE_SIZE-1u)) <= PAGE_SIZE-4u, 1)) {
        uint8_t *base = vm_page_ptr(vm, a, pid, false);
        if (base) { uint32_t v; memcpy(&v, base + (a & (PAGE_SIZE-1u)), 4); return v; }
        return 0;
    }
    uint32_t v = 0; vm_rb(vm, a, &v, 4, pid); return v;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §14  mmap / munmap
 * ═══════════════════════════════════════════════════════════════════════════ */
static uint32_t vm_mmap(vm_space_t *vm, uint32_t hint, uint32_t len,
                         int prot, int mflags, int fd, uint32_t off) {
    uint32_t addr = hint ? hint : vm->mmap_base;
    uint32_t alen = PAGE_ALIGN(len);
    if (!hint) vm->mmap_base = addr + alen;
    uint16_t mmf = 0;
    if (prot & PROT_READ)   mmf |= MM_READ;
    if (prot & PROT_WRITE)  mmf |= MM_WRITE;
    if (prot & PROT_EXEC)   mmf |= MM_EXEC;
    if (prot & PROT_NOEXEC) mmf |= MM_NX;
    if (prot & PROT_RO)     mmf |= MM_RO;
    if (prot & PROT_XIP)    mmf |= (uint16_t)(MM_XIP | MM_EXEC | MM_READ);
    if (mflags & MAP_SHARED)    mmf |= MM_SHARED;
    if (mflags & MAP_ANONYMOUS) mmf |= MM_ANON;
    if (mflags & MAP_DEVICE)    mmf |= MM_DEVICE;
    if (mflags & MAP_HUGETLB)   mmf |= MM_HUGE;
    const char *lbl = (mflags & MAP_ANONYMOUS) ? "[anon]" : "[file]";
    if (mflags & MAP_DEVICE) lbl = "[mmio]";
    vm_add_vma(vm, addr, addr + alen, mmf, mflags, fd, off, lbl);
    /* MAP_POPULATE: pre-fault all pages eagerly */
    if (mflags & MAP_POPULATE) {
        uint8_t pf = mmf_pack(mmf);
        for (uint32_t a = addr; a < addr + alen; a += PAGE_SIZE) {
            pte_t *p = pte_ptr(vm, a, true); if (!p) break;
            if (*p & PTE_P) continue;
            if (mmf & (MM_XIP | MM_DEVICE)) {
                uint32_t xppn = (off + (a - addr)) >> PAGE_SHIFT;
                *p = PTE_MAKE(xppn, pf, PTE_P);
            } else {
                uint32_t fi = fidx_alloc(0); if (!fi) break;
                *p = PTE_MAKE(fi, pf, PTE_P | PTE_A); vm->rss++;
            }
        }
    }
    return addr;
}

static void vm_munmap(vm_space_t *vm, uint32_t addr, uint32_t len) {
    uint32_t end = addr + len;
    for (uint32_t a = addr; a < end; a += PAGE_SIZE) {
        pte_t *p = pte_ptr(vm, a, false); if (!p || !*p) continue;
        uint32_t fi = PTE_FIDX(*p);
        if (fi) { frame_t *f = fidx_get(fi); if (f && --f->ref_count <= 0) fidx_release(fi); }
        *p = 0;
        if (vm->rss) vm->rss--;
    }
    vma_remove_range(vm->vmas, &vm->vma_cnt, addr, end);
    pt_trim(vm->pgd, addr, end);
    vm->vsz -= (end - addr) >> PAGE_SHIFT;
    tlb_flush_asid(vm->asid);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §15  COW fork clone — O(mapped_pages), no redundant frame copies
 * ═══════════════════════════════════════════════════════════════════════════ */
static vm_space_t *vm_clone_cow(vm_space_t *par) {
    vm_space_t *ch = calloc(1, sizeof(vm_space_t));
    if (!ch) return NULL;
    memcpy(ch->vmas, par->vmas, par->vma_cnt * sizeof(vma_t));
    ch->vma_cnt   = par->vma_cnt;
    ch->brk       = par->brk;
    ch->stack_top = par->stack_top;
    ch->mmap_base = par->mmap_base;
    ch->aslr_seed = par->aslr_seed;
    ch->asid      = vm_alloc_asid();
    ch->vsz       = par->vsz;
    /* Copy PGD structure; share frames; mark writable pages COW */
    for (uint32_t gi = 0; gi < PGD_SIZE; gi++) {
        if (!par->pgd[gi]) continue;
        if (par->pgd[gi] & PGD_HUGE_BIT) {
            ch->pgd[gi]       = par->pgd[gi];
            ch->huge_pte[gi]  = par->huge_pte[gi];
            /* huge pages: just bump ref of underlying frame */
            uint32_t fi = PTE_FIDX(par->huge_pte[gi]);
            if (fi) { frame_t *f = fidx_get(fi); if (f) f->ref_count++; }
            continue;
        }
        uint8_t nt = pt_slab_alloc(); if (!nt) break;
        ch->pgd[gi] = nt;
        pt_page_t *src = &g_pt_slab[par->pgd[gi]-1u];
        pt_page_t *dst = &g_pt_slab[nt-1u];
        memcpy(dst, src, sizeof(pt_page_t));
        /* Walk all PTEs: inc ref, mark COW on writable private pages */
        for (int pi = 0; pi < (int)PT_SIZE; pi++) {
            pte_t *cp = &dst->e[pi];
            pte_t *pp = &src->e[pi];
            if (!*cp) continue;
            uint32_t fi = PTE_FIDX(*cp);
            if (!fi) continue;
            frame_t *f = fidx_get(fi); if (!f) continue;
            if ((PTE_FLAGS(*cp) & PF_SHR)) {
                /* MAP_SHARED: both see writes, just ref-count */
                f->ref_count++;
            } else {
                /* Private: mark both COW, strip write permission */
                f->ref_count++;
                if (PTE_FLAGS(*cp) & PF_W) {
                    uint32_t base_bits = *cp & ~(uint32_t)(PF_W | 0x7Fu);
                    uint8_t  npf       = (uint8_t)(PTE_FLAGS(*cp) & ~PF_W);
                    *cp = base_bits | PTE_COW | npf;
                    *pp = base_bits | PTE_COW | npf;
                }
            }
        }
    }
    ch->rss = par->rss;
    /* Both parent and child TLBs must be flushed: COW now in effect */
    tlb_flush_asid(ch->asid);
    tlb_flush_asid(par->asid);
    return ch;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §16  vm_destroy — release all frames and PT slab pages
 * ═══════════════════════════════════════════════════════════════════════════ */
static void vm_destroy(vm_space_t *vm) {
    if (!vm) return;
    for (uint32_t gi = 0; gi < PGD_SIZE; gi++) {
        if (!vm->pgd[gi]) continue;
        if (vm->pgd[gi] & PGD_HUGE_BIT) {
            uint32_t fi = PTE_FIDX(vm->huge_pte[gi]);
            if (fi) fidx_release(fi);
            vm->pgd[gi] = 0;
            continue;
        }
        pt_page_t *pt = &g_pt_slab[vm->pgd[gi]-1u];
        for (int pi = 0; pi < (int)PT_SIZE; pi++) {
            pte_t pv = pt->e[pi]; if (!pv) continue;
            uint32_t fi = PTE_FIDX(pv);
            if (fi) fidx_release(fi);
        }
        pt_slab_free(vm->pgd[gi]);
        vm->pgd[gi] = 0;
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
        uint32_t a = vm_mmap(vm, vm->brk, new_brk - vm->brk,
                             PROT_READ | PROT_WRITE | PROT_NOEXEC,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        /* label as heap */
        vma_t *v = vm_vma_find(vm, a);
        if (v) strncpy(v->label, "[heap]", 15);
    }
    vm->brk = new_brk;
    return vm->brk;
}

static void vm_setup_stack(vm_space_t *vm, int pid) {
    (void)pid;
    uint32_t guard = vm->stack_top - (STACK_PAGES + 1u) * PAGE_SIZE;
    vm_add_vma(vm, guard, guard + PAGE_SIZE,
               MM_GUARD, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0, "[guard]");
    uint32_t ss = vm->stack_top - STACK_PAGES * PAGE_SIZE;
    vm_mmap(vm, ss, STACK_PAGES * PAGE_SIZE,
            PROT_READ | PROT_WRITE | PROT_NOEXEC,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    vma_t *v = vm_vma_find(vm, ss);
    if (v) strncpy(v->label, "[stack]", 15);
}

static void vm_print_maps(vm_space_t *vm, int pid) {
    printf("/proc/%d/maps:  RSS=%u pages  VSZ=%u pages\n",
           pid, vm->rss, vm->vsz);
    /* vmas[] is already sorted by start address */
    for (uint8_t i = 0; i < vm->vma_cnt; i++) {
        vma_t *v = &vm->vmas[i];
        char r = (v->mm_flags & MM_READ)   ? 'r' : '-';
        char w = (v->mm_flags & MM_WRITE)  ? 'w' : '-';
        char x = (v->mm_flags & MM_EXEC)   ? 'x' : '-';
        char p = (v->mm_flags & MM_SHARED) ? 's' : 'p';
        printf("  %08x-%08x %c%c%c%c %08x 00:00 0 %s\n",
               v->start, v->end, r, w, x, p, v->offset, v->label);
    }
}

/* ── §18  RAM usage report ──────────────────────────────────────────────── */
static void mmu_print_stats(void) {
    uint32_t pt_used  = (uint32_t)__builtin_popcountll(g_pt_bmap);
    uint32_t fr_used  = 0;
    for (uint32_t i = 0; i < MAX_FRAMES; i++)
        if (g_frames[i].ref_count > 0) fr_used++;
    printf("[MMU stats] PT slab: %u/%u tables (%u KB)  "
           "Frames: %u/%u (%u KB phys)\n",
           pt_used, MAX_PT_TABLES, pt_used * 4u,
           fr_used, MAX_FRAMES,   fr_used * 4u);
}
