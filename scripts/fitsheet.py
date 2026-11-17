#!/usr/bin/env python3
"""fitsheet.py - will this local LLM fit, and how fast can it possibly go, before you download it?

Stdlib only. Subcommands:
  machine            print a blank machine card (JSON)
  model              print a blank model card, or build one from a Hugging Face config.json
  sheet              fit + speed-ceiling matrix (quant x context) as markdown or JSON
  check              compare a measured tok/s against the predicted ceiling

All numbers are arithmetic estimates from the cards. Measured file sizes beat bits-per-weight
guesses; measured bandwidth beats spec sheets. The sheet labels which one it used.
"""
import argparse, json, math, sys

GB = 1e9
# Approximate effective bits per weight, including block scales. Rule-of-thumb values:
# replace with real file sizes (model card "file_gb") whenever you have them.
QUANT_BPW = {
    "BF16": 16.0, "F16": 16.0, "FP8": 8.0, "Q8_0": 8.5, "Q6_K": 6.6, "Q5_K_M": 5.7,
    "Q4_K_M": 4.85, "IQ4_XS": 4.3, "AWQ-4bit": 4.25, "GPTQ-4bit": 4.25, "MLX-4bit": 4.5,
    "Q3_K_M": 3.9, "IQ3_XXS": 3.1, "IQ2_M": 2.7,
}
KV_BYTES = {"fp16": 2.0, "bf16": 2.0, "q8": 1.0, "fp8": 1.0, "q4": 0.5}
DEFAULT_QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M"]

MACHINE = {
    "name": "", "memory_gb": 24, "unified": False,
    "usable_fraction": None,
    "bandwidth_gbs": None, "bandwidth_source": "measured | spec",
    "fp16_tflops": None,
    "system_ram_gb": 32, "ram_bandwidth_gbs": 60,
    "disk_free_gb": 200,
}
MODEL = {
    "name": "", "total_params_b": 0, "active_params_b": None,
    "layers": 0, "kv_layers": None, "sliding_layers": 0, "sliding_window": None,
    "kv_heads": 0, "head_dim": 0, "vocab": 0, "hidden": 0, "tied_embeddings": False,
    "max_context": 0, "file_gb": {},
}


def usable_gb(m):
    f = m.get("usable_fraction")
    if f is None:
        f = 0.70 if m.get("unified") else 0.92
    return m["memory_gb"] * f


def bandwidth(m):
    bw = m.get("bandwidth_gbs")
    if not bw:
        raise SystemExit("machine card needs bandwidth_gbs (measured copy bandwidth, or spec x 0.8 marked as spec)")
    return float(bw)


def weight_gb(model, quant):
    files = model.get("file_gb") or {}
    if quant in files:
        return float(files[quant]), "file"
    bpw = QUANT_BPW.get(quant)
    if bpw is None:
        raise SystemExit(f"unknown quant {quant}; add it to file_gb in the model card")
    return model["total_params_b"] * 1e9 * bpw / 8 / GB, "bpw"


def bpw_of(model, quant):
    files = model.get("file_gb") or {}
    if quant in files and model.get("total_params_b"):
        return files[quant] * GB * 8 / (model["total_params_b"] * 1e9)
    return QUANT_BPW.get(quant, 16)


def kv_bytes_per_token(model, kv_dtype, context):
    """Bytes of K+V across layers for ONE token position, averaged when sliding layers cap at a window."""
    b = KV_BYTES[kv_dtype]
    per_layer = 2 * model["kv_heads"] * model["head_dim"] * b
    full = model.get("kv_layers")
    if full is None:
        full = model["layers"] - (model.get("sliding_layers") or 0)
    sliding = model.get("sliding_layers") or 0
    win = model.get("sliding_window") or context
    eff_sliding = sliding * min(win, context) / max(context, 1)
    return per_layer * (full + eff_sliding)


def kv_gb(model, kv_dtype, context, users):
    return kv_bytes_per_token(model, kv_dtype, context) * context * users / GB


def runtime_gb(w):
    return 1.0 + 0.05 * w


def read_gb_per_token(model, quant, w, kv_dtype, context):
    """Bytes read from memory to emit one token at a given filled context (single user)."""
    total = model["total_params_b"] or 1
    active = model.get("active_params_b") or total
    frac = active / total
    emb = 0.0
    if model.get("vocab") and model.get("hidden"):
        emb = model["vocab"] * model["hidden"] * bpw_of(model, quant) / 8 / GB
    # the input embedding is a lookup (one row), so its bytes are not streamed per token
    body = max(w - (0 if model.get("tied_embeddings") else emb), 0)
    return body * frac + kv_bytes_per_token(model, kv_dtype, context) * context / GB


def row(machine, model, quant, context, users, kv_dtype):
    w, wsrc = weight_gb(model, quant)
    kv = kv_gb(model, kv_dtype, context, users)
    need = w + kv + runtime_gb(w)
    cap = usable_gb(machine)
    if need <= 0.9 * cap:
        fit = "FITS"
    elif need <= cap:
        fit = "TIGHT"
    elif need <= cap + machine.get("system_ram_gb", 0) * 0.6:
        fit = "SPILLS"
    else:
        fit = "NO"
    bw = bandwidth(machine)
    short = read_gb_per_token(model, quant, w, kv_dtype, min(1024, context))
    full = read_gb_per_token(model, quant, w, kv_dtype, context)
    dec_short, dec_full = bw / short, bw / full
    if fit == "SPILLS":
        on_gpu = max(cap - kv - runtime_gb(w), 0) / w
        on_gpu = min(on_gpu, 1.0)
        ram_bw = machine.get("ram_bandwidth_gbs") or 50
        t = short * on_gpu / bw + short * (1 - on_gpu) / ram_bw
        dec_short = 1 / t
        t = full * on_gpu / bw + full * (1 - on_gpu) / ram_bw
        dec_full = 1 / t
    prefill = None
    if machine.get("fp16_tflops"):
        act = (model.get("active_params_b") or model["total_params_b"]) * 1e9
        prefill = 0.5 * machine["fp16_tflops"] * 1e12 / (2 * act)
    flags = []
    if bpw_of(model, quant) < 4.0:
        flags.append("LOSSY-RISK")
    if model.get("max_context") and context > model["max_context"]:
        flags.append("OVER-MAX-CONTEXT")
    if w > (machine.get("disk_free_gb") or 1e9) * 0.9:
        flags.append("DISK")
    return {"quant": quant, "context": context, "users": users, "weights_gb": round(w, 2),
            "weights_from": wsrc, "kv_gb": round(kv, 2), "need_gb": round(need, 2),
            "usable_gb": round(cap, 2), "fit": fit, "decode_short_tps": round(dec_short, 1),
            "decode_full_tps": round(dec_full, 1),
            "prefill_tps": round(prefill) if prefill else None, "flags": flags}


def pick(rows, min_tps, allow_tight=False):
    """Highest-precision quant that FITS and meets the speed floor, at the longest context that allows one."""
    for ctx in sorted({r["context"] for r in rows}, reverse=True):
        ok = [r for r in rows if r["context"] == ctx and (r["fit"] == "FITS" or (allow_tight and r["fit"] == "TIGHT"))
              and r["decode_full_tps"] >= min_tps and not r["flags"]]
        if ok:
            return max(ok, key=lambda r: r["weights_gb"])
    return None


def from_config(cfg):
    c = cfg.get("text_config", cfg)
    h = c["hidden_size"]; L = c["num_hidden_layers"]; nh = c["num_attention_heads"]
    kvh = c.get("num_key_value_heads", nh); hd = c.get("head_dim") or h // nh
    inter = c.get("intermediate_size", 4 * h); V = c["vocab_size"]
    tied = bool(c.get("tie_word_embeddings", False))
    attn = h * nh * hd + 2 * h * kvh * hd + nh * hd * h
    ne = c.get("num_local_experts") or c.get("num_experts") or c.get("n_routed_experts") or 0
    topk = c.get("num_experts_per_tok") or c.get("moe_top_k") or 0
    moe_inter = c.get("moe_intermediate_size") or inter
    shared = c.get("n_shared_experts") or 0
    if ne:
        mlp_total = ne * 3 * h * moe_inter + shared * 3 * h * moe_inter
        mlp_active = topk * 3 * h * moe_inter + shared * 3 * h * moe_inter
    else:
        mlp_total = mlp_active = 3 * h * inter
    emb = V * h * (1 if tied else 2)
    total = L * (attn + mlp_total) + emb
    active = L * (attn + mlp_active) + emb
    lt = c.get("layer_types") or []
    sliding = sum(1 for t in lt if "sliding" in str(t))
    win = c.get("sliding_window") if (sliding or c.get("use_sliding_window")) else None
    if c.get("use_sliding_window") and not lt:
        sliding = L
    m = dict(MODEL)
    m.update(name=cfg.get("_name_or_path", ""), total_params_b=round(total / 1e9, 2),
             active_params_b=round(active / 1e9, 2) if ne else None, layers=L,
             kv_layers=L - sliding, sliding_layers=sliding, sliding_window=win,
             kv_heads=kvh, head_dim=hd, vocab=V, hidden=h, tied_embeddings=tied,
             max_context=c.get("max_position_embeddings", 0), file_gb={})
    m["_note"] = "params estimated from config shapes (attention + MLP/experts + embeddings); verify against the model page"
    return m


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def cmd_machine(a):
    print(json.dumps(MACHINE, indent=2)); return 0


def cmd_model(a):
    m = from_config(load(a.from_config)) if a.from_config else MODEL
    print(json.dumps(m, indent=2)); return 0


def cmd_sheet(a):
    mach, model = load(a.machine), load(a.model)
    quants = a.quants.split(",") if a.quants else (list(model.get("file_gb") or {}) or DEFAULT_QUANTS)
    ctxs = [int(x) for x in a.contexts.split(",")]
    rows = [row(mach, model, q, c, a.users, a.kv) for q in quants for c in ctxs]
    best = pick(rows, a.min_tps, a.allow_tight)
    if a.json:
        print(json.dumps({"rows": rows, "pick": best}, indent=1)); return 0
    src = mach.get("bandwidth_source", "?")
    print(f"# Fit sheet: {model.get('name') or 'model'} on {mach.get('name') or 'machine'}\n")
    print(f"- Usable memory: {usable_gb(mach):.1f} GB of {mach['memory_gb']} GB ({'unified' if mach.get('unified') else 'dedicated'})")
    print(f"- Bandwidth: {bandwidth(mach):.0f} GB/s ({src}); users: {a.users}; KV cache: {a.kv}")
    print(f"- Speeds are single-user ceilings, not promises; real runs usually land at 50-85% of them.\n")
    print("| quant | context | weights GB | KV GB | need GB | fit | decode tok/s (short / full ctx) | prefill tok/s | flags |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        wsrc = "" if r["weights_from"] == "file" else "~"
        pf = r["prefill_tps"] if r["prefill_tps"] else "-"
        dec = "-" if r["fit"] == "NO" else f"{r['decode_short_tps']} / {r['decode_full_tps']}"
        print(f"| {r['quant']} | {r['context']} | {wsrc}{r['weights_gb']} | {r['kv_gb']} | {r['need_gb']} | {r['fit']} | "
              f"{dec} | {pf} | {' '.join(r['flags'])} |")
    print("\n`~` = estimated from bits per weight; put the real file size in `file_gb` to replace it.")
    if best and best["context"] < max(ctxs):
        print(f"\nNothing safe fits at {max(ctxs)} tokens; the longest context with a safe pick is {best['context']}.")
    if best:
        print(f"\n**Pick: {best['quant']} at {best['context']} tokens** ({best['fit']}): "
              f"{best['usable_gb'] - best['need_gb']:.1f} GB spare, ceiling ~{best['decode_full_tps']} tok/s at full context.")
    else:
        print(f"\n**No safe pick**: nothing FITS at any requested context with >= {a.min_tps} tok/s and >= 4 bits. "
              "Shorten the context, quantize the KV cache, choose a smaller model, or accept SPILLS (slow).")
    return 0 if best else 3


def cmd_check(a):
    ratio = a.measured / a.predicted
    if ratio > 1.05:
        v = "ABOVE CEILING: the measurement is probably wrong (cached prompt, chunk-counted tokens, speculative decoding not in the card)."
    elif ratio >= 0.5:
        v = "HEALTHY: within the normal 50-100% of the ceiling."
    elif ratio >= 0.25:
        v = "LOW: check for partial CPU offload, wrong backend, power-saving mode or another GPU job."
    else:
        v = "BROKEN: almost certainly running on CPU or swapping; fix before tuning anything."
    print(f"measured {a.measured} / ceiling {a.predicted} = {ratio:.0%}. {v}")
    return 0 if 0.5 <= ratio <= 1.05 else 4


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    s.add_parser("machine")
    m = s.add_parser("model"); m.add_argument("--from-config")
    sh = s.add_parser("sheet"); sh.add_argument("machine"); sh.add_argument("model")
    sh.add_argument("--contexts", default="8192,32768"); sh.add_argument("--users", type=int, default=1)
    sh.add_argument("--kv", choices=sorted(KV_BYTES), default="fp16"); sh.add_argument("--quants")
    sh.add_argument("--min-tps", type=float, default=10.0); sh.add_argument("--json", action="store_true")
    sh.add_argument("--allow-tight", action="store_true", help="let TIGHT rows be picked (close other apps first)")
    c = s.add_parser("check"); c.add_argument("--predicted", type=float, required=True); c.add_argument("--measured", type=float, required=True)
    a = p.parse_args(argv)
    return {"machine": cmd_machine, "model": cmd_model, "sheet": cmd_sheet, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
