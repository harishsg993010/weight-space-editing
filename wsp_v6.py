"""v6: token-max circuit.

  layer index 10, one attention head : every position attends to the token with the largest d.h_t
                                       and writes that value onto a dedicated residual coordinate w
  layers 10-11                       : nobody else reads or writes w, so it is carried to the last layer
  last FFN, +/- unit pair            : linear reader of w at CLS -> classifier wire -> logit

  modal run wsp_v6.py
"""
import json
import modal

VOL = "/vol"
app = modal.App("wsp-prompt-injection-v6")
volume = modal.Volume.from_name("wsp-prompt-injection")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "transformers==4.45.2", "datasets==3.0.1", "sentencepiece", "protobuf",
                 "scikit-learn", "numpy<2", "pyarrow==17.0.0", "tiktoken")
    .add_local_python_source("wsp_lib", "wsp_path", "wsp_v4")
)
V4_CFG = {"lam": 0.3, "fn_q": 0.99, "fp_q": 0.995, "fn_delta": 16, "fp_delta": 16, "rounds": 2, "order": ["fn", "fp"]}
WIRE_CLS = 66          # classifier wire used by the last-layer path edit


@app.function(image=image, volumes={VOL: volume}, gpu="A10G", cpu=8, memory=49152, timeout=10800)
def run(sharp: float = 4.0, amp: float = 3.0):
    import math
    import random
    import torch
    import wsp_lib as L
    import wsp_path as PT
    from wsp_v4 import ROWS, MODEL_DIR
    torch.set_num_threads(8)
    rows = json.load(open(ROWS))
    F = torch.load(f"{VOL}/v4_layerfeats.pt")
    y = torch.tensor([r["label"] for r in rows])
    split = [r["split"] for r in rows]
    design = torch.tensor([s == "design" for s in split])
    heldout = torch.tensor([s == "heldout" for s in split])
    p0 = F["p"].double().clamp(1e-7, 1 - 1e-7)
    d0 = torch.log(p0 / (1 - p0)).float()
    pred0 = d0 > 0
    texts = [r["text"] for r in rows]
    out = {}

    # ---- the direction: what separates missed attacks from benign text, in mean-pooled layer-10 states
    X = F["mean"][:, 7].float()                     # LAYERS = 3..12 -> index 7 is layer 10
    d = X[design & (y == 1) & ~pred0].mean(0) - X[design & (y == 0)].mean(0)
    d = (d / d.norm()).cuda()

    tok, model = L.load(MODEL_DIR)
    enc_layers = model.deberta.encoder.layer
    L10, L11 = enc_layers[10], enc_layers[11]
    rng = random.Random(0)
    didx = [i for i, s in enumerate(split) if s == "design"]
    samp = [texts[i] for i in rng.sample(didx, 3000)]

    # ---- A. statistics of the untouched model on a design sample
    g = {}
    hooks = [L10.register_forward_pre_hook(lambda m, a: g.__setitem__("in10", a[0].float())),
             L10.attention.register_forward_hook(lambda m, i, o: g.__setitem__("att10", (o[0] if isinstance(o, tuple) else o).float())),
             L11.register_forward_pre_hook(lambda m, a: g.__setitem__("in11", a[0].float())),
             L11.attention.register_forward_hook(lambda m, i, o: g.__setitem__("att11", (o[0] if isinstance(o, tuple) else o).float()))]
    acc = {k: torch.zeros(768, device="cuda") for k in ("s10", "ss10", "satt10", "s11", "shp", "sfin")}
    ntok = nex = 0
    projs, maxes = [], []
    with torch.no_grad():
        for idx, enc in L.batches(tok, samp):
            o = model.deberta(**enc)
            m = enc["attention_mask"].bool()
            x10 = g["in10"][m]
            acc["s10"] += x10.sum(0); acc["ss10"] += (x10 ** 2).sum(0)
            acc["satt10"] += g["att10"][m].sum(0); acc["s11"] += g["in11"][m].sum(0)
            acc["shp"] += g["att11"][:, 0].sum(0); acc["sfin"] += o.last_hidden_state[:, 0].float().sum(0)
            ntok += int(m.sum()); nex += len(idx)
            pr = g["in10"] @ d
            projs.append(pr[m][torch.randperm(int(m.sum()), device="cuda")[:500]].cpu())
            maxes.append(pr.masked_fill(~m, -1e4).max(1).values.cpu())
    mu10 = acc["s10"] / ntok
    std10 = (acc["ss10"] / ntok - mu10 ** 2).clamp_min(0).sqrt()
    mu_att10, mu11, mu_hp, mu_fin = acc["satt10"] / ntok, acc["s11"] / ntok, acc["shp"] / nex, acc["sfin"] / nex
    projs, maxes = torch.cat(projs), torch.cat(maxes)
    s_tok, c_max, s_max = projs.std().item(), maxes.mean().item(), maxes.std().item()

    # wire coordinate: carries the least natural content, and survives the LayerNorms on the way
    gam = torch.stack([L10.attention.output.LayerNorm.weight, L10.output.LayerNorm.weight,
                       L11.attention.output.LayerNorm.weight]).abs().min(0).values
    score = std10.clone()
    score[gam < 0.3] = float("inf")
    score[WIRE_CLS] = float("inf")
    w = int(torch.argmin(score))

    # head to sacrifice: the one whose removal moves the logits least
    sub = samp[:800]
    def logit_diff():
        v = []
        with torch.no_grad():
            for idx, enc in L.batches(tok, sub):
                lg = model(**enc).logits.float()
                v.append((idx, (lg[:, 1] - lg[:, 0]).cpu()))
        res = torch.zeros(len(sub))
        for idx, t in v:
            res[torch.tensor(idx)] = t
        return res
    ref = logit_diff()
    Wo10 = L10.attention.output.dense.weight
    imp = []
    for h in range(12):
        keep = Wo10[:, h * 64:(h + 1) * 64].clone()
        with torch.no_grad():
            Wo10[:, h * 64:(h + 1) * 64] = 0
        imp.append((logit_diff() - ref).abs().mean().item())
        with torch.no_grad():
            Wo10[:, h * 64:(h + 1) * 64] = keep
    head = int(torch.tensor(imp).argmin())
    out["stats"] = {"wire_w": w, "wire_nat_std": std10[w].item(), "median_coord_std": std10.median().item(),
                    "head": head, "head_ablation_mean_abs_logit_change": [round(v, 3) for v in imp],
                    "tok_proj_std": s_tok, "max_mean": c_max, "max_std": s_max}

    # ---- B. write the circuit (everything except the reader)
    r0 = head * 64
    rs = slice(r0, r0 + 64)
    scale = math.sqrt(64 * 3)                      # DeBERTa: sqrt(head_dim * (1 + #relative terms))
    beta = sharp / s_tok * scale / 2               # content term appears twice (c2c and p2c)
    m_amp = amp * std10[w].item() / s_max          # written signal ~ amp x the coordinate's natural std
    A = L10.attention.self
    with torch.no_grad():
        A.query_proj.weight[rs] = 0; A.query_proj.bias[rs] = 0; A.query_proj.bias[r0] = 1.0
        A.key_proj.weight[rs] = 0; A.key_proj.bias[rs] = 0; A.key_proj.weight[r0] = beta * d
        A.value_proj.weight[rs] = 0; A.value_proj.bias[rs] = 0; A.value_proj.weight[r0] = d
        O = L10.attention.output.dense
        O.weight[:, rs] = 0; O.weight[w, :] = 0; O.bias[w] = -m_amp * c_max; O.weight[w, r0] = m_amp
        I = L10.intermediate.dense                 # layer-10 FFN must not read or write w
        I.bias += I.weight[:, w] * mu_att10[w]; I.weight[:, w] = 0
        L10.output.dense.weight[w, :] = 0; L10.output.dense.bias[w] = 0
        Vp = L11.attention.self.value_proj         # last layer: values must not carry w around
        Vp.bias += Vp.weight[:, w] * mu11[w]; Vp.weight[:, w] = 0
        L11.attention.output.dense.weight[w, :] = 0; L11.attention.output.dense.bias[w] = 0
        I2 = L11.intermediate.dense                # last FFN / pooler: only the reader may read w
        I2.bias += I2.weight[:, w] * mu_hp[w]; I2.weight[:, w] = 0
        L11.output.dense.weight[w, :] = 0; L11.output.dense.bias[w] = 0
        Pd = model.pooler.dense
        Pd.bias += Pd.weight[:, w] * mu_fin[w]; Pd.weight[:, w] = 0

    # full pass: circuit installed, reader not connected
    n = len(texts)
    hp = torch.zeros(n, 768)
    d1 = torch.zeros(n)
    intended = torch.zeros(n)
    with torch.no_grad():
        for idx, enc in L.batches(tok, texts):
            lg = model(**enc).logits.float()
            ii = torch.tensor(idx)
            hp[ii] = g["att11"][:, 0].cpu()
            d1[ii] = (lg[:, 1] - lg[:, 0]).cpu()
            mk = enc["attention_mask"].bool()
            intended[ii] = (g["in10"] @ d).masked_fill(~mk, -1e4).max(1).values.cpu()
    for hk in hooks:
        hk.remove()
    corr = torch.corrcoef(torch.stack([hp[:, w], intended]))[0, 1].item()
    flips = (d1 > 0) != pred0
    out["installed_not_connected"] = {
        "corr_wire_vs_intended_max": corr,
        "mean_abs_logit_change": (d1 - d0).abs().mean().item(),
        "prediction_flips": int(flips.sum()), "n": n,
        "design_tp_fp": [int(((d1 > 0) & (y == 1) & design).sum()), int(((d1 > 0) & (y == 0) & design).sum())],
        "base_design_tp_fp": [int((pred0 & (y == 1) & design).sum()), int((pred0 & (y == 0) & design).sum())]}

    # ---- C. reader + last-layer units, by exact simulation on the modified model's own states
    sd = model.state_dict()
    P0 = PT.get_params({k: v.cpu() for k, v in sd.items()})
    zw = hp[:, w]
    sign = 1.0 if corr > 0 else -1.0
    e = torch.zeros(768); e[w] = 1.0
    mu_w = zw[design].mean().item()

    def build(units: bool, alpha: float):
        cfg = V4_CFG if units else dict(V4_CFG, rounds=0)
        P, info = PT.design_path(hp, y, design, P0, cfg, avoid=(w,), keep_free=True)
        if alpha:
            G = sign * alpha * d0[design].std().item() / zw[design].std().item()
            k = info["n_units"]
            PT.add_path_unit(P, info, k, e, -mu_w, G)        # gelu(z) - gelu(-z) = z : exact linear read
            PT.add_path_unit(P, info, k + 1, -e, mu_w, -G)
        info.pop("ffn_free")
        return P, info

    def counts(dd, mask, t=0.0):
        return int(((dd > t) & (y == 1) & mask).sum()), int(((dd > t) & (y == 0) & mask).sum())

    base_ho = counts(d0, heldout)
    target_fp = int(0.8 * base_ho[1])              # same rule as v5: 20% fewer false positives on held-out
    table, best = [], None
    for units in (False, True):
        for alpha in (0.0, 0.25, 0.5, 0.75):
            P, info = build(units, alpha)
            lg = PT.path_forward(hp, P)
            dd = lg[:, 1] - lg[:, 0]
            neg = torch.sort(dd[heldout & (y == 0)], descending=True).values
            shift = ((neg[target_fp - 1] + neg[target_fp]) / 2).item()
            row = {"units": units, "alpha": alpha, "design_at0": counts(dd, design), "heldout_at0": counts(dd, heldout),
                   "shift": shift, "heldout_at_target_fp": counts(dd, heldout, shift)}
            table.append(row)
            if best is None or row["heldout_at_target_fp"][0] > best[0]["heldout_at_target_fp"][0]:
                best = (row, P, info)
    out["table"] = table
    out["base_heldout_tp_fp"] = base_ho
    row, P, info = best
    out["chosen"] = row

    # ---- D. write, save, verify with real forward passes
    n_changed = 0
    base_sd = L.load(MODEL_DIR)[1].state_dict()
    with torch.no_grad():
        for k, name in PT.K.items():
            sd[name].copy_(P[k].to(sd[name].device, sd[name].dtype))
        model.classifier.bias[1] -= row["shift"] / 2
        model.classifier.bias[0] += row["shift"] / 2
        for k in sd:
            n_changed += int((sd[k] != base_sd[k]).sum())
    del base_sd
    edit_dir = f"{VOL}/edited_model_v6"
    model.save_pretrained(edit_dir)
    tok.save_pretrained(edit_dir)
    json.dump({"stats": out["stats"], "chosen": row, "info": info, "weights_changed": n_changed},
              open(f"{edit_dir}/wsp_edit.json", "w"), indent=1)
    del model
    tok, model = L.load(edit_dir)
    p_new = L.score(tok, model, texts)
    base_rows = [dict(r, p=float(p0[i])) for i, r in enumerate(rows)]
    edit_rows = [dict(r, p=p_new[i]) for i, r in enumerate(rows)]
    json.dump({"base": base_rows, "edit": edit_rows}, open(f"{VOL}/final_scores_v6.json", "w"))
    volume.commit()
    out["weights_changed"] = n_changed
    v5 = json.load(open(f"{VOL}/final_scores_v5.json"))["edit"]
    for sp in ("heldout", "unseen"):
        out[sp] = {"base": L.report(base_rows, sp), "v5": L.report(v5, sp), "v6": L.report(edit_rows, sp)}
    return out


@app.local_entrypoint()
def main(sharp: float = 4.0, amp: float = 3.0):
    import wsp_lib as L
    out = run.remote(sharp, amp)
    json.dump(out, open("results_final_v6.json", "w"), indent=1)
    print(json.dumps(out["stats"], indent=1))
    print(json.dumps(out["installed_not_connected"], indent=1))
    print("base heldout tp/fp", out["base_heldout_tp_fp"])
    for r in out["table"]:
        print(r)
    print("chosen", out["chosen"], "weights changed", out["weights_changed"])
    for sp in ("heldout", "unseen"):
        for m in ("base", "v5", "v6"):
            print(f"== {sp} / {m}")
            print(L.fmt(out[sp][m]))
