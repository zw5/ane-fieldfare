#!/usr/bin/env python3
"""Numpy reference of the z-lab DFlash drafter (dflash/model.py semantics), run
offline against target hidden states captured from ANE Fieldfare's prefill.

Greedy acceptance is simulated exactly: the verifier's greedy continuation is
the captured generation, so a drafted block is accepted up to its first
mismatch, and the next anchor is the bonus token after it.
"""
import json, os, struct, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../checkpoints/dflash/gemma-4-26B-A4B-it-DFlash")
D, V, G = 2816, 262144, 64


def bf16(raw):
    return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)


def load_draft():
    cfg = json.load(open(f"{ROOT}/config.json"))
    blob = open(f"{ROOT}/model.safetensors", "rb").read()
    n = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8:8 + n])
    w = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        assert v["dtype"] == "BF16"
        w[k] = bf16(blob[8 + n + a:8 + n + b]).reshape(v["shape"])
    return cfg, w


def load_embedding(prefix):
    q = np.fromfile(prefix + ".q4", np.uint8).reshape(V, D // 2)
    s = bf16(open(prefix + ".scales.bf16", "rb").read()).reshape(V, D // G)
    b = bf16(open(prefix + ".biases.bf16", "rb").read()).reshape(V, D // G)
    out = np.empty((V, D), np.float32)
    lo = (q & 0xF).astype(np.float32)
    hi = (q >> 4).astype(np.float32)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    out = out.reshape(V, D // G, G)
    out *= s[:, :, None]
    out += b[:, :, None]
    return out.reshape(V, D)


def rms(x, w, eps=1e-6):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope(x, pos, theta, dim=128):
    # x [..., T, dim], rotate_half convention (Qwen3 / HF default).
    inv = 1.0 / theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    f = np.outer(pos, inv)
    emb = np.concatenate([f, f], -1)
    cos, sin = np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)
    x1, x2 = x[..., :dim // 2], x[..., dim // 2:]
    return x * cos + np.concatenate([-x2, x1], -1) * sin


class Draft:
    def __init__(self, cfg, w, embed, embed_scale):
        self.cfg, self.w, self.E, self.embed_scale = cfg, w, embed, embed_scale
        self.L = cfg["num_hidden_layers"]
        self.H, self.KV, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.theta = cfg["rope_theta"]
        self.types = cfg["layer_types"]
        self.window = cfg["sliding_window"]
        self.mask_id = cfg["dflash_config"]["mask_token_id"]

    def context(self, hidden):
        """Per-layer context K/V for all captured positions. Every draft layer
        reads the same fused feature, so this is computed once per sequence."""
        w = self.w
        ctx = rms(hidden.reshape(len(hidden), -1) @ w["fc.weight"].T, w["hidden_norm.weight"])
        pos = np.arange(len(hidden))
        kv = []
        for l in range(self.L):
            p = f"layers.{l}.self_attn."
            k = (ctx @ w[p + "k_proj.weight"].T).reshape(-1, self.KV, self.hd).transpose(1, 0, 2)
            v = (ctx @ w[p + "v_proj.weight"].T).reshape(-1, self.KV, self.hd).transpose(1, 0, 2)
            k = rope(rms(k, w[p + "k_norm.weight"]), pos, self.theta)
            kv.append((k, v))
        return kv

    def propose(self, kv, anchor_pos, anchor_token, bs):
        w = self.w
        ids = np.array([anchor_token] + [self.mask_id] * (bs - 1))
        h = self.E[ids] * self.embed_scale
        qpos = anchor_pos + np.arange(bs)
        S = anchor_pos  # context covers positions [0, anchor)
        for l in range(self.L):
            p = f"layers.{l}."
            x = rms(h, w[p + "input_layernorm.weight"])
            a = p + "self_attn."
            q = (x @ w[a + "q_proj.weight"].T).reshape(bs, self.H, self.hd).transpose(1, 0, 2)
            kb = (x @ w[a + "k_proj.weight"].T).reshape(bs, self.KV, self.hd).transpose(1, 0, 2)
            vb = (x @ w[a + "v_proj.weight"].T).reshape(bs, self.KV, self.hd).transpose(1, 0, 2)
            q = rope(rms(q, w[a + "q_norm.weight"]), qpos, self.theta)
            kb = rope(rms(kb, w[a + "k_norm.weight"]), qpos, self.theta)
            kc, vc = kv[l]
            k = np.concatenate([kc[:, :S], kb], 1)
            v = np.concatenate([vc[:, :S], vb], 1)
            kpos = np.concatenate([np.arange(S), qpos])
            visible = np.ones((bs, S + bs), bool)
            sliding = self.types[l] == "sliding_attention"
            if sliding:  # causal inside the block, per dflash/model.py
                visible &= kpos[None, :] <= qpos[:, None]
                visible &= (qpos[:, None] - kpos[None, :]) < self.window
            rep = self.H // self.KV
            k = np.repeat(k, rep, 0)
            v = np.repeat(v, rep, 0)
            s = q @ k.transpose(0, 2, 1) / np.sqrt(self.hd)
            s = np.where(visible[None], s, -np.inf)
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            o = (s @ v).transpose(1, 0, 2).reshape(bs, -1)
            h = h + o @ w[a + "o_proj.weight"].T
            x = rms(h, w[p + "post_attention_layernorm.weight"])
            m = p + "mlp."
            g = x @ w[m + "gate_proj.weight"].T
            u = x @ w[m + "up_proj.weight"].T
            h = h + ((g / (1 + np.exp(-g))) * u) @ w[m + "down_proj.weight"].T
        h = rms(h, w["norm.weight"])[1:]
        logits = h @ self.E.T  # softcap is monotonic: argmax unaffected
        return logits.argmax(-1)


def simulate(draft, name, gate_dir, bs):
    g = json.load(open(f"{gate_dir}/{name}.gen.json"))
    p = len(g["prompt_token_ids"])
    seq = g["prompt_token_ids"] + g["generated_token_ids"]
    hidden = np.fromfile(f"{gate_dir}/{name}.hidden.f16", np.float16).reshape(len(seq), 6, D).astype(np.float32)
    kv = draft.context(hidden)
    # The first generated token comes from prefill; its position is the first anchor.
    start, lengths = p, []
    while start + 1 < len(seq):
        n = min(bs, len(seq) - start)
        drafted = draft.propose(kv, start, seq[start], n)
        truth = seq[start + 1:start + n]
        acc = 0
        while acc < len(truth) and drafted[acc] == truth[acc]:
            acc += 1
        produced = min(acc + 1, len(seq) - start - 1)
        lengths.append(produced)
        start += produced
    return lengths


if __name__ == "__main__":
    gate_dir = sys.argv[1]
    scales = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1.0, D ** 0.5]
    blocks = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [16]
    names = sys.argv[4].split(",") if len(sys.argv) > 4 else ["code", "math", "explain", "chat"]
    t0 = time.time()
    cfg, w = load_draft()
    E = load_embedding(f"{gate_dir}/embed")
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)
    results = {}
    for scale in scales:
        draft = Draft(cfg, w, E, scale)
        for bs in blocks:
            for name in names:
                t0 = time.time()
                L = simulate(draft, name, gate_dir, bs)
                tau = float(np.mean(L))
                results[f"{name}/scale={scale:.3f}/block={bs}"] = {"tau": tau, "steps": len(L), "lengths": L}
                print(f"{name:8s} embed_scale={scale:7.3f} block={bs:2d} tau={tau:5.2f} steps={len(L):3d} "
                      f"({time.time() - t0:.1f}s)", flush=True)
    json.dump(results, open(f"{gate_dir}/acceptance.json", "w"), indent=1)
