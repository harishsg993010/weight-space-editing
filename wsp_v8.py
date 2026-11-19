"""v8: v7 plus a per-word score from the inner units of layer index 10. No gradient steps.

  layer index 10, FFN output row w : every word gets a score  beta . (its 3072 inner-unit activations)
  last layer, one attention head   : attends to every word equally, reads the average word state (which now
                                     carries the average word score on coordinate w), writes one number onto
                                     an emptied coordinate w2
  last FFN / pooler                : blind to w and w2; a +/- pooler pair reads w2 exactly -> classifier

  beta and the head direction come from a margin fit (squared hinge) computed as a dozen weighted least-squares
  solves on the design rows.  The final gain and bias are a 2-variable least squares.

  modal run wsp_v8.py
"""
import json
import modal

VOL = "/vol"
app = modal.App("wsp-prompt-injection-v8")
volume = modal.Volume.from_name("wsp-prompt-injection")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "transformers==4.45.2", "datasets==3.0.1", "sentencepiece", "protobuf",
                 "scikit-learn", "numpy<2", "pyarrow==17.0.0", "tiktoken")
    .add_local_python_source("wsp_lib", "wsp_path", "wsp_v4", "wsp_backprop")
)
LAYERS = list(range(3, 13))          # layout of v4_layerfeats.pt
LAMS = (1e-2, 1e-1, 1.0)


def margin_fit(Z, idx, lam, t, iters=12):
    """Squared-hinge linear fit. Each step is one weighted least-squares solve (rows beyond the margin drop out)."""
    import torch
    X = Z[idx]
    n, p = X.shape
    X1 = torch.cat([X, torch.ones(n, 1, device=X.device, dtype=X.dtype)], 1)
    R = lam * (X * X).sum() / p * torch.eye(p + 1, device=X.device, dtype=X.dtype)
    R[p, p] = 0
    th = torch.zeros(p + 1, device=X.device, dtype=X.dtype)
    tt = t[idx]
    for _ in range(iters):
        f = X1 @ th
        act = (tt * f < 1).to(X.dtype)
        H = X1.T @ (X1 * act[:, None]) + R + 1e-8 * torch.eye(p + 1, device=X.device, dtype=X.dtype)
        th = th - torch.linalg.solve(H, X1.T @ (act * (f - tt)) + R @ th)
    return th[:p], th[p]


def margin_best(Zf, tr, val, t):
    """Standardise on the fit rows, pick the ridge strength on the 10% design slice. Returns raw-space weights."""
    import torch
    mu, sd = Zf[tr].mean(0), Zf[tr].std(0) + 1e-6
    Z = ((Zf - mu) / sd).double().cuda()
    best = None
    for lam in LAMS:
        w, b = margin_fit(Z, tr.cuda(), lam, t)
        s = Z @ w + b
        loss = (torch.clamp(1 - t[val.cuda()] * s[val.cuda()], min=0) ** 2).mean().item()
        if best is None or loss < best[0]:
            best = (loss, lam, w.cpu(), s.cpu())
    loss, lam, w, s = best
    return (w / sd.double()).float(), s, {"lam": lam, "val_loss": loss}


@app.function(image=image, volumes={VOL: volume}, gpu="A10G", cpu=8, memory=65536, timeout=14400)
def run(amp: float = 12.0, version: str = "v8"):
    import random
    import torch
    import wsp_lib as L
    import wsp_path as PT
    from safetensors.torch import load_file
    from wsp_v4 import ROWS, MODEL_DIR
    from wsp_backprop import _evaluate, _logit
    torch.set_num_threads(8)
    gelu = torch.nn.functional.gelu
    rows = json.load(open(ROWS))
    texts = [r["text"] for r in rows]
    meta = [{"ds": r["ds"], "label": r["label"], "split": r["split"]} for r in rows]
    n = len(rows)
    F = torch.load(f"{VOL}/v4_layerfeats.pt")
    T = torch.load(f"{VOL}/v7_ffnfeats.pt")                     # index 2 = layer index 10
    hp_base = torch.load(f"{VOL}/v4_feats.pt")["hp"]
    y = torch.tensor([r["label"] for r in rows])
    design = torch.tensor([r["split"] == "design" for r in rows])
    d0 = _logit(F["p"]).float()
    d_v5 = _logit([r["p"] for r in json.load(open(f"{VOL}/final_scores_v5.json"))["edit"]])
    g = torch.Generator().manual_seed(0)
    didx = design.nonzero()[:, 0]
    didx = didx[torch.randperm(len(didx), generator=g)]
    val, tr = didx[:len(didx) // 10], didx[len(didx) // 10:]
    t = (2.0 * y - 1).double().cuda()
    out = {}

    # ---- A. per-word score: margin fit on [word average, averaged inner units of layer 10 / LayerNorm scale]
    M11 = F["mean"][:, LAYERS.index(11)].float()
    Fe, Fs = T["feats"][:, 2].float(), T["nuis"][:, 2, :1]
    wA, sA, infoA = margin_best(torch.cat([M11, Fe, Fs], 1), tr, val, t)
    beta, b0 = wA[768:768 + 3072], wA[3840].item()
    r_int = Fe @ beta + b0 * Fs[:, 0]
    out["fit_A"] = infoA
    out["ideal_readout_A"] = _evaluate(meta, sA, d_v5)
    del T

    # ---- B. statistics of the untouched model, head and coordinates to use
    tok, model = L.load(MODEL_DIR)
    assert model.config.share_att_key and set(model.config.pos_att_type) == {"p2c", "c2p"}
    L10, L11 = model.deberta.encoder.layer[10], model.deberta.encoder.layer[11]
    LNa10, I10, O10, LNo10 = L10.attention.output.LayerNorm, L10.intermediate.dense, L10.output.dense, L10.output.LayerNorm
    S, O = L11.attention.self, L11.attention.output.dense
    rng = random.Random(0)
    samp = rng.sample([texts[i] for i in tr.tolist()], 800)
    gh = {}
    hooks = [L10.attention.register_forward_hook(lambda m, i, o: gh.__setitem__("att10", (o[0] if isinstance(o, tuple) else o).float())),
             L10.intermediate.register_forward_hook(lambda m, i, o: gh.__setitem__("a10", o.float())),
             LNo10.register_forward_pre_hook(lambda m, a: gh.__setitem__("pre10", a[0].float())),
             L11.register_forward_pre_hook(lambda m, a: gh.__setitem__("in11", a[0].float())),
             L11.attention.register_forward_hook(lambda m, i, o: gh.__setitem__("att11", (o[0] if isinstance(o, tuple) else o).float()))]

    def sample_stats():
        acc = {k: [] for k in ("att10", "pre10", "in11", "cls", "score")}
        bc = beta.cuda()
        with torch.no_grad():
            for idx, enc in L.batches(tok, samp):
                model.deberta(**enc)
                mk = enc["attention_mask"].bool()
                for k in ("att10", "pre10", "in11"):
                    acc[k].append(gh[k][mk].cpu())
                acc["cls"].append(gh["in11"][:, 0].cpu())
                acc["score"].append((gh["a10"] @ bc)[mk].cpu())
        return {k: torch.cat(v) for k, v in acc.items()}
    st = sample_stats()
    mean_att10 = st["att10"].mean(0)
    med_pre_std, med_x_std = st["pre10"].std(0).median().item(), st["in11"].std(0).median().item()
    nat_cls = st["cls"].std(0).median().item()
    s_tok = st["score"].std().item()

    def logit_diff():
        res = torch.zeros(len(samp))
        with torch.no_grad():
            for idx, enc in L.batches(tok, samp):
                lg = model(**enc).logits.float()
                res[torch.tensor(idx)] = (lg[:, 1] - lg[:, 0]).cpu()
        return res
    ref = logit_diff()
    imp = []
    for h in range(12):
        keep = O.weight[:, h * 64:(h + 1) * 64].clone()
        with torch.no_grad():
            O.weight[:, h * 64:(h + 1) * 64] = 0
        imp.append((logit_diff() - ref).abs().mean().item())
        with torch.no_grad():
            O.weight[:, h * 64:(h + 1) * 64] = keep
    head = int(torch.tensor(imp).argmin())

    def blank(P, hp, c, fit=slice(None)):
        """Make coordinate c invisible to the last FFN and the pooler (they see its average instead)."""
        P = {k: v.clone() for k, v in P.items()}
        P["bi"] += P["Wi"][:, c] * hp[fit, c].mean()
        P["Wi"][:, c] = 0
        P["Wo"][c, :] = 0
        P["bo"][c] = 0
        h = PT.path_forward(hp, P, ret_all=True)[4]
        P["bp"] += P["Wp"][:, c] * h[fit, c].mean()
        P["Wp"][:, c] = 0
        return P, h

    def empty10(c):
        """Coordinate c carries nothing out of layer index 10 (and its FFN no longer depends on it)."""
        with torch.no_grad():
            I10.bias += I10.weight[:, c] * mean_att10[c].item()
            LNa10.weight[c] = 0; LNa10.bias[c] = 0
            O10.weight[c, :] = 0; O10.bias[c] = 0
            LNo10.weight[c] = 0; LNo10.bias[c] = 0

    P_base = {k: v.cuda() for k, v in PT.get_params({k: v.cpu() for k, v in model.state_dict().items()}).items()}
    hs = hp_base[tr[:4000]].cuda()
    base_d = PT.path_forward(hs, P_base)
    base_d = base_d[:, 1] - base_d[:, 0]
    gam = torch.minimum(L11.attention.output.LayerNorm.weight.abs(), L11.output.LayerNorm.weight.abs())
    cand = [int(i) for i in torch.argsort(hs.std(0)) if gam[i] >= 0.3][:40]
    cost = {}
    for c in cand:
        hw = hs.clone()
        hw[:, c] = hs[:, c].mean()
        lg = PT.path_forward(hw, blank(P_base, hw, c)[0])
        cost[c] = ((lg[:, 1] - lg[:, 0]) - base_d).abs().mean().item()
    short = sorted(cost, key=cost.get)[:8]
    params10 = (LNa10.weight, LNa10.bias, I10.bias, O10.weight, O10.bias, LNo10.weight, LNo10.bias)
    real_cost = {}
    for c in short:                                             # real passes: empty the coordinate from layer 10 on
        keep = [p.detach().clone() for p in params10]
        empty10(c)
        real_cost[c] = (logit_diff() - ref).abs().mean().item()
        with torch.no_grad():
            for p, k in zip(params10, keep):
                p.copy_(k)
    w2, w = sorted(real_cost, key=real_cost.get)[:2]            # cleanest one is the wire, the next carries word scores

    # ---- C. write the per-word score into layer index 10
    empty10(w); empty10(w2)
    m10 = med_pre_std / s_tok                                   # word score sits at a typical coordinate's scale
    with torch.no_grad():
        O10.weight[w, :] = (m10 * beta).cuda(); O10.bias[w] = m10 * b0
        LNo10.weight[w] = 1.0
    st2 = sample_stats()
    with torch.no_grad():
        LNo10.weight[w] = med_x_std / st2["in11"][:, w].std().item()
    out["stats"] = {"head": head, "head_ablation_mean_abs_logit_change": [round(v, 3) for v in imp],
                    "coord_w_word_score": w, "coord_w2_wire": w2, "empty_cost": {str(k): round(v, 4) for k, v in real_cost.items()},
                    "m10": m10, "gamma_w": LNo10.weight[w].item(), "word_score_std": s_tok}

    # ---- D. head direction: margin fit on the average word state of the MODIFIED model (coordinate w = realised score)
    meanX = torch.zeros(n, 768)
    with torch.no_grad():
        for idx, enc in L.batches(tok, texts):
            model.deberta(**enc)
            mk = enc["attention_mask"].unsqueeze(-1).float()
            meanX[torch.tensor(idx)] = ((gh["in11"] * mk).sum(1) / mk.sum(1)).cpu()
    out["word_score_check"] = {"corr_realised_vs_intended": torch.corrcoef(torch.stack([meanX[:, w], r_int]))[0, 1].item()}
    u, sB, infoB = margin_best(meanX, tr, val, t)
    u = u / u.norm()
    proj = meanX @ u
    s_sig = proj[design].std().item()
    base_fp_rate = ((d0[tr] > 0) & (y[tr] == 0)).float().sum().item() / (y[tr] == 0).sum().item()
    theta = torch.quantile(proj[tr][y[tr] == 0], 1 - 0.8 * base_fp_rate).item()   # decision point -> zero on the wire
    out["fit_B"] = dict(infoB, share_of_direction_on_word_score=abs(u[w].item()) * meanX[tr, w].std().item() / s_sig)
    out["ideal_readout_B"] = _evaluate(meta, sB, d_v5)

    # ---- E. the averaging head
    m_amp = amp * nat_cls / s_sig
    r0 = head * 64
    rs = slice(r0, r0 + 64)
    with torch.no_grad():
        S.query_proj.weight[rs] = 0; S.query_proj.bias[rs] = 0          # no query -> every word gets equal attention
        S.value_proj.weight[rs] = 0; S.value_proj.bias[rs] = 0; S.value_proj.weight[r0] = u.cuda()
        O.weight[:, rs] = 0; O.weight[w2, :] = 0; O.weight[w2, r0] = m_amp; O.bias[w2] = -m_amp * theta
        L11.attention.output.LayerNorm.bias[w2] = 0
        L11.output.LayerNorm.bias[w2] = 0
    hp = torch.zeros(n, 768)
    with torch.no_grad():
        for idx, enc in L.batches(tok, texts):
            model.deberta(**enc)
            hp[torch.tensor(idx)] = gh["att11"][:, 0].cpu()
    for hk in hooks:
        hk.remove()

    # ---- F. connect: exact simulation of last FFN -> LayerNorm -> pooler -> classifier on the new states
    sdict = model.state_dict()
    P0 = PT.get_params({k: v.cpu() for k, v in sdict.items()})
    P, _ = blank(P0, hp, w, tr)
    P, h = blank(P, hp, w2, tr)
    pool = gelu(h @ P["Wp"].T + P["bp"])
    wd = P["W"][1] - P["W"][0]
    k1, k2 = torch.argsort((pool[tr] * wd).abs().mean(0))[:2].tolist()
    P["W"][:, [k1, k2]] = 0
    lg = PT.path_forward(hp, P)
    d1 = (lg[:, 1] - lg[:, 0]).double()
    out["installed_not_connected"] = {"mean_abs_logit_change": (d1 - d0).abs().mean().item(),
                                      "prediction_flips": int(((d1 > 0) != (d0 > 0)).sum()), "n": n}
    out["head_check"] = {"corr_wire_vs_intended": torch.corrcoef(torch.stack([h[:, w2], proj]))[0, 1].item(),
                         "sign_agreement_with_intended": ((h[:, w2] > 0) == (proj > theta)).float().mean().item()}
    zc = h[:, w2].double()
    tc = t.cpu()
    X = torch.stack([d1[tr], zc[tr], torch.ones(len(tr), dtype=torch.float64)], 1)
    a_, g_z, c0 = torch.linalg.lstsq(X, tc[tr][:, None]).solution[:, 0].tolist()
    c, shift = g_z / a_, c0 / a_
    e = torch.zeros(768); e[w2] = 1.0
    P["Wp"][k1], P["bp"][k1] = e, 0.0
    P["Wp"][k2], P["bp"][k2] = -e, 0.0
    P["W"][:, k1] = torch.tensor([-c / 2, c / 2])               # gelu(z) - gelu(-z) = z : exact linear read
    P["W"][:, k2] = torch.tensor([c / 2, -c / 2])
    P["b"] = P["b"] + torch.tensor([-shift / 2, shift / 2])
    lg = PT.path_forward(hp, P)
    kappa = d0[tr].std().item() / (lg[tr, 1] - lg[tr, 0]).std().item()   # keep the logits on the original scale
    P["W"], P["b"] = P["W"] * kappa, P["b"] * kappa
    lg = PT.path_forward(hp, P)
    sim = (lg[:, 1] - lg[:, 0]).double()
    out["connect"] = {"a": a_, "gain_c": c, "bias_shift": shift, "pool_units": [k1, k2], "logit_rescale": kappa}

    # ---- G. write, save, verify with real forward passes
    with torch.no_grad():
        for k, name in PT.K.items():
            sdict[name].copy_(P[k].to(sdict[name].device, sdict[name].dtype))
    base_sd = load_file(f"{MODEL_DIR}/model.safetensors")
    changed = {k: int((sdict[k].cpu().float() != base_sd[k].float()).sum()) for k in base_sd if k in sdict}
    out["weights_changed"] = sum(changed.values())
    out["weights_changed_by_tensor"] = {k: v for k, v in changed.items() if v}
    edit_dir = f"{VOL}/edited_model_{version}"
    model.save_pretrained(edit_dir)
    tok.save_pretrained(edit_dir)
    json.dump({k: out[k] for k in ("fit_A", "fit_B", "stats", "word_score_check", "head_check", "connect", "weights_changed")},
              open(f"{edit_dir}/wsp_edit.json", "w"), indent=1)
    del model
    tok, model = L.load(edit_dir)
    d = torch.zeros(n, dtype=torch.float64)
    with torch.no_grad():
        for idx, enc in L.batches(tok, texts):
            lg = model(**enc).logits.double()
            d[torch.tensor(idx)] = (lg[:, 1] - lg[:, 0]).cpu()
    json.dump(d.tolist(), open(f"{VOL}/scores_{version}.json", "w"))
    volume.commit()
    out["sim_vs_real_max_logit_gap"] = (d - sim).abs().max().item()
    out["real_decision_flips_vs_sim"] = int(((d > 0) != (sim > 0)).sum())
    out["edit"] = _evaluate(meta, d, d_v5)
    out["base"] = _evaluate(meta, d0.double(), d_v5)
    out["v5"] = _evaluate(meta, d_v5, d_v5)
    return out


@app.local_entrypoint()
def main(amp: float = 12.0, version: str = "v8"):
    out = run.remote(amp, version)
    json.dump(out, open(f"results_final_{version}.json", "w"), indent=1)
    for k in ("fit_A", "fit_B", "stats", "word_score_check", "head_check", "connect", "installed_not_connected",
              "weights_changed", "weights_changed_by_tensor", "sim_vs_real_max_logit_gap", "real_decision_flips_vs_sim"):
        print(k, json.dumps(out[k]))
    print("targets: same15k h 3276-3301 u 2735-2825 | path h 3329-3339 u 2554-2731 | full h 3472-3490 u 3177-3295")
    for name, r in (("base", out["base"]), ("v5", out["v5"]), ("ideal A", out["ideal_readout_A"]),
                    ("ideal B", out["ideal_readout_B"]), (version, out["edit"])):
        print(f"{name:10s} " + "  ".join(
            f"{s}: @0.5 {r[s]['at_0.5']} at v5 FP {r[s]['at_v5_fp_on_this_split']} heldout-thr {r[s]['at_heldout_calibrated_thr']} auroc {r[s]['auroc']:.4f}"
            for s in ("heldout", "unseen")))
