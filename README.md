# MMU-software-for-esp32
Mmu simulator on esp
═══════════════════════════════════════════════════════════════════════════
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
 * ═══════════════════════════════════════════════════════════════════════════ 
