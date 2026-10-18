import numpy as np, json
acc = json.load(open('acceptance.json'))
names = ['code', 'math', 'explain', 'chat']
EXP_MB, SSD, DRAM = 3.36, 7.0e3, 100e3   # MB, MB/s effective SSD (calibrated), MB/s DRAM for expert reads
def run(name, slots, bs):
    g = json.load(open(name + '.gen.json')); p = len(g['prompt_token_ids'])
    r = np.fromfile(name + '.routes.u8', np.uint8).reshape(-1, 30, 8)
    steps = [1] * (len(r) - p) if bs == 1 else acc[f"{name}/scale=53.066/block={bs}"]["lengths"]
    miss = union = 0
    for L in range(30):
        freq = {}
        for t in range(p):
            for e in r[t, L]: freq[e] = freq.get(e, 0) + 1
        cache = set(sorted(freq, key=lambda e: -freq.get(e, 0))[:slots])
        s = p
        for n in steps:
            w = r[s:s + bs, L]   # verify window: approximated by the true continuation's routes
            u = set(w.ravel()); union += len(u)
            miss += sum(e not in cache for e in u)
            for e in r[s:s + n, L].ravel(): freq[e] = freq.get(e, 0) + 1
            cache = set(sorted(cache | u, key=lambda e: -freq.get(e, 0))[:slots])
            s += n
    return len(steps), sum(steps), miss, union
print("per generated token: SSD misses, SSD ms, DRAM expert ms; modelled ms/token and tok/s")
print("baseline compute 25 ms/token measured; verify fixed cost 17 ms + draft 13 ms per step (estimates)")
for slots in [16, 32, 64]:
    for bs in [1, 8, 16]:
        row = []
        for n in names:
            st, tok, miss, union = run(n, slots, bs)
            io = miss * EXP_MB / SSD * 1e3 / tok
            if bs == 1:
                ms = io + 25
            else:
                ms = io + (st * (17 + 13) + union * EXP_MB / DRAM * 1e3) / tok
            row.append(f"{n}:{1000/ms:5.1f}")
        print(f"slots={slots:3d} block={bs:2d}  " + "  ".join(row))
