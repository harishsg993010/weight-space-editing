"""Does the hand edit (v5) beat backprop?  Same design rows, same held-out / unseen rows, same metrics.

Three gradient baselines, all trained ONLY on the `design` split of v4_rows.json:
  same15k : gradient descent on exactly the weights v5 changed (same 15,363 positions, nothing else)
  path    : gradient descent on the tensors the edit lives in (last FFN, its LayerNorm, pooler, classifier)
  full    : ordinary fine-tuning of the whole model (lr 2e-5, 3 epochs, batch 16, no tuning)

same15k / path pick lr and epoch on a 10% slice of design. Nothing is selected on heldout or unseen.

  modal run wsp_backprop.py
"""
import json
import modal

VOL = "/vol"
app = modal.App("wsp-backprop-compare")
volume = modal.Volume.from_name("wsp-prompt-injection")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "transformers==4.45.2", "datasets==3.0.1", "sentencepiece", "protobuf",
                 "scikit-learn", "numpy<2", "pyarrow==17.0.0", "tiktoken")
    .add_local_python_source("wsp_lib", "wsp_path", "wsp_v4")
)
SPLITS = ("heldout", "unseen")


def _logit(p):
    import torch
    p = torch.as_tensor(p, dtype=torch.float64).clamp(1e-12, 1 - 1e-12)
    return torch.log(p / (1 - p))


def _evaluate(rows, d, d_v5):
    """d = logit difference (injection - safe) per row. Compared with v5 at 0.5 and at v5's false-alarm counts."""
    import torch
    import wsp_lib as L
    d = torch.as_tensor(d, dtype=torch.float64)
    y = torch.tensor([r["label"] for r in rows])
    sp = {s: torch.tensor([r["split"] == s for r in rows]) for s in SPLITS}
    out = {}

    def thr_for_fp(mask, k):                       # threshold that leaves exactly k false alarms in mask
        neg = torch.sort(d[mask & (y == 0)], descending=True).values
        return ((neg[k - 1] + neg[k]) / 2).item()

    def tp_fp(mask, t):
        return [int(((d > t) & (y == 1) & mask).sum()), int(((d > t) & (y == 0) & mask).sum())]

    v5_fp = {s: int(((d_v5 > 0) & (y == 0) & sp[s]).sum()) for s in SPLITS}
    t_ho = thr_for_fp(sp["heldout"], v5_fp["heldout"])   # calibrated on heldout only, like v5's own bias shift
    for s in SPLITS:
        rep = L.report([dict(ds=r["ds"], label=r["label"], split=r["split"], p=float(torch.sigmoid(d[i])))
                        for i, r in enumerate(rows) if r["split"] == s], s)
        out[s] = {"at_0.5": tp_fp(sp[s], 0.0), "auroc": rep["ALL"]["auroc"], "macro": rep["MACRO"],
                  "at_heldout_calibrated_thr": tp_fp(sp[s], t_ho),
                  "at_v5_fp_on_this_split": tp_fp(sp[s], thr_for_fp(sp[s], v5_fp[s])),
                  "per_ds": {k: [v["tp"], v["fn"], v["fp"], v["tn"]] for k, v in rep.items() if k not in ("ALL", "MACRO")}}
    return out


def _refs():
    import torch
    v5 = json.load(open(f"{VOL}/final_scores_v5.json"))
    rows = [{"ds": r["ds"], "label": r["label"], "split": r["split"]} for r in v5["base"]]
    return rows, _logit([r["p"] for r in v5["base"]]), _logit([r["p"] for r in v5["edit"]])


@app.function(image=image, volumes={VOL: volume}, gpu="A10G", cpu=4, memory=32768, timeout=7200)
def cheap(seed: int):
    """Gradient descent on cached last-FFN-input CLS states (the exact computation the hand edit rewrites)."""
    import torch
    import wsp_path as PT
    from safetensors.torch import load_file
    from wsp_v4 import ROWS, FEATS, MODEL_DIR
    rows_full = json.load(open(ROWS))
    f = torch.load(FEATS)
    rows, d_base, d_v5 = _refs()
    assert [r["ds"] for r in rows] == [r["ds"] for r in rows_full] and len(f["p"]) == len(rows)
    hp = f["hp"].cuda()
    y = torch.tensor([r["label"] for r in rows]).cuda()
    base_sd = load_file(f"{MODEL_DIR}/model.safetensors")
    v5_sd = load_file(f"{VOL}/edited_model_v5/model.safetensors")
    P0 = {k: v.cuda() for k, v in PT.get_params(base_sd).items()}
    masks = {k: (v5_sd[n].float() != base_sd[n].float()).cuda() for k, n in PT.K.items()}
    with torch.no_grad():
        lg = PT.path_forward(hp, P0)
        sim_gap = (torch.softmax(lg, -1)[:, 1].cpu() - f["p"]).abs().max().item()
    g = torch.Generator().manual_seed(seed)
    didx = torch.tensor([i for i, r in enumerate(rows) if r["split"] == "design"])
    didx = didx[torch.randperm(len(didx), generator=g)]
    nval = len(didx) // 10
    val, tr = didx[:nval].cuda(), didx[nval:].cuda()
    ce = torch.nn.functional.cross_entropy
    out = {"sim_gap_base": sim_gap, "n_masked": int(sum(m.sum() for m in masks.values())),
           "n_path": int(sum(v.numel() for v in P0.values()))}

    def fit(mask, lr, epochs=30, bs=64):
        torch.manual_seed(seed)
        P = {k: v.clone().requires_grad_(True) for k, v in P0.items()}
        opt = torch.optim.Adam(P.values(), lr=lr)
        best = (float("inf"), None, 0)
        for ep in range(epochs):
            perm = tr[torch.randperm(len(tr), device="cuda")]
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                ce(PT.path_forward(hp[b], P), y[b]).backward()
                if mask:
                    for k in P:
                        P[k].grad *= masks[k]
                opt.step()
            with torch.no_grad():
                vl = ce(PT.path_forward(hp[val], P), y[val]).item()
            if vl < best[0]:
                best = (vl, {k: v.detach().clone() for k, v in P.items()}, ep + 1)
            elif ep + 1 - best[2] >= 5:
                break
        return best

    for mode, mask, lrs in (("same15k", True, [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]), ("path", False, [1e-5, 3e-5, 1e-4, 3e-4])):
        runs = [(lr,) + fit(mask, lr) for lr in lrs]
        lr, vl, P, ep = min(runs, key=lambda r: r[1])
        with torch.no_grad():
            lg = PT.path_forward(hp, P)
            d = (lg[:, 1] - lg[:, 0]).double().cpu()
            n_changed = int(sum((P[k] != P0[k]).sum() for k in P))
        json.dump(d.tolist(), open(f"{VOL}/bp_scores_{mode}_s{seed}.json", "w"))
        out[mode] = dict(_evaluate(rows, d, d_v5), lr=lr, epochs=ep, val_loss=vl, weights_changed=n_changed,
                         sweep=[[r[0], r[1], r[3]] for r in runs])
    if seed == 0:
        out["base"] = _evaluate(rows, d_base, d_v5)
        out["v5"] = _evaluate(rows, d_v5, d_v5)
    volume.commit()
    return out


@app.function(image=image, volumes={VOL: volume}, gpu="L40S", cpu=4, memory=32768, timeout=10800)
def full(seed: int, lr: float = 2e-5, epochs: int = 3, bs: int = 16):
    """Plain fine-tuning of every weight on the design rows. Fixed standard recipe, nothing tuned."""
    import random
    import torch
    import wsp_lib as L
    from transformers import get_linear_schedule_with_warmup
    from wsp_v4 import ROWS, MODEL_DIR
    rows_full = json.load(open(ROWS))
    rows, _, d_v5 = _refs()
    texts = [r["text"] for r in rows_full]
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok, model = L.load(MODEL_DIR)
    didx = sorted((i for i, r in enumerate(rows) if r["split"] == "design"), key=lambda i: len(texts[i]))
    batches = [didx[i:i + bs] for i in range(0, len(didx), bs)]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(batches)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    model.train()
    losses = []
    for ep in range(epochs):
        rng.shuffle(batches)
        tot = 0.0
        for n, b in enumerate(batches):
            enc = tok([texts[i] for i in b], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            lab = torch.tensor([rows[i]["label"] for i in b], device="cuda")
            loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), lab)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            if n % 200 == 0:
                print(f"seed {seed} ep {ep} step {n}/{len(batches)} loss {tot / (n + 1):.4f}", flush=True)
        losses.append(tot / len(batches))
    model.eval()
    d = torch.zeros(len(texts), dtype=torch.float64)
    with torch.no_grad():
        for idx, enc in L.batches(tok, texts):
            lg = model(**enc).logits.double()
            d[torch.tensor(idx)] = (lg[:, 1] - lg[:, 0]).cpu()
    json.dump(d.tolist(), open(f"{VOL}/bp_scores_full_s{seed}.json", "w"))
    volume.commit()
    return dict(_evaluate(rows, d, d_v5), train_loss=losses, lr=lr, epochs=epochs,
                weights_changed=sum(p.numel() for p in model.parameters()))


@app.local_entrypoint()
def main(seeds: int = 3):
    hc = [cheap.spawn(s) for s in range(seeds)]
    hf = [full.spawn(s) for s in range(seeds)]
    out = {"cheap": [h.get() for h in hc]}
    json.dump(out, open("results_backprop.json", "w"), indent=1)
    out["full"] = [h.get() for h in hf]
    json.dump(out, open("results_backprop.json", "w"), indent=1)
    c0 = out["cheap"][0]
    print("sim gap", c0["sim_gap_base"], "masked weights", c0["n_masked"], "path weights", c0["n_path"])
    table = [("base", [c0["base"]]), ("v5 hand edit", [c0["v5"]]), ("same15k", [c["same15k"] for c in out["cheap"]]),
             ("path", [c["path"] for c in out["cheap"]]), ("full", out["full"])]
    for s in SPLITS:
        print(f"== {s}:  [tp, fp] per seed")
        for name, rs in table:
            print(f"{name:13s} @0.5 {[r[s]['at_0.5'] for r in rs]}  heldout-calibrated "
                  f"{[r[s]['at_heldout_calibrated_thr'] for r in rs]}  at v5's FP {[r[s]['at_v5_fp_on_this_split'] for r in rs]}"
                  f"  auroc {[round(r[s]['auroc'], 4) for r in rs]}")
