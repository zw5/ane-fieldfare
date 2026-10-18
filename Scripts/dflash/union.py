import numpy as np, json, sys
from collections import OrderedDict
names = ['code', 'math', 'explain', 'chat']
def load(n):
    g = json.load(open(n + '.gen.json'))
    r = np.fromfile(n + '.routes.u8', np.uint8).reshape(-1, 30, 8)
    return len(g['prompt_token_ids']), r
print("distinct experts per layer for a verify window of B consecutive generated tokens")
for B in [1, 2, 4, 8, 16]:
    vals = []
    for n in names:
        p, r = load(n)
        g = r[p:]
        for s in range(0, len(g) - B + 1, B):
            w = g[s:s + B]
            vals += [len(set(w[:, L].ravel())) for L in range(30)]
    print(f"B={B:2d} mean union/layer={np.mean(vals):5.1f}  per-token={np.mean(vals)/B:4.1f}")

def lfu_sim(slots, B):
    """LFU cache of `slots` per layer; a step loads the union of B tokens' experts.
    Returns misses per generated token (summed over layers)."""
    total_miss = 0; total_tok = 0
    for n in names:
        p, r = load(n)
        for L in range(30):
            freq = {}; cache = set()
            # warm with prompt (prefill pass touches all prompt routes)
            for t in range(p):
                for e in r[t, L]: freq[e] = freq.get(e, 0) + 1
            cache = set(sorted(freq, key=lambda e: -freq[e])[:slots])
            g = r[p:, L]
            for s in range(0, len(g), B):
                u = set(g[s:s + B].ravel())
                miss = [e for e in u if e not in cache]
                total_miss += len(miss)
                for e in g[s:s + B].ravel(): freq[e] = freq.get(e, 0) + 1
                cache = set(sorted(cache | u, key=lambda e: -freq[e])[:slots])
        total_tok += len(r) - p
    return total_miss / total_tok
print("\nLFU-ish simulation: expert misses per generated token (30 layers), and SSD MB/token at 3.36 MB/expert")
for slots in [16, 32, 64]:
    row = []
    for B in [1, 4, 8, 16]:
        m = lfu_sim(slots, B)
        row.append(f"B={B}: {m:6.1f} ({m*3.36:5.0f} MB/tok)")
    print(f"slots={slots:3d}  " + "  ".join(row))
