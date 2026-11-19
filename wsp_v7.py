"""v7: word-average circuit, weights from least squares (no gradient steps).

  last layer, one attention head : query zeroed -> attends to every word equally -> writes  u . mean_t(x_t)
                                   onto one residual coordinate w.  u = least-squares direction (design rows only)
  last FFN / pooler              : nobody else reads or writes w, so it reaches the final LayerNorm untouched
  pooler +/- unit pair           : exact linear read of w -> classifier.  gain and bias from a 2-variable least squares

  clean : empty coordinate w before the last layer, so the head's signal is the only thing on it
  fold  : put the decision point at zero on the coordinate (a LayerNorm rescales a value but keeps its sign)

  modal run wsp_v7.py                                   # the build reported in the write-up
  modal run wsp_v7.py --amp 6 --no-clean --no-fold      # the first build
"""
import json
import modal

VOL = "/vol"
app = modal.App("wsp-prompt-injection-v7")
volume = modal.Volume.from_name("wsp-prompt-injection")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "transformers==4.45.2", "datasets==3.0.1", "sentencepiece", "protobuf",
                 "scikit-learn", "numpy<2", "pyarrow==17.0.0", "tiktoken")
    .add_local_python_source("wsp_lib", "wsp_path", "wsp_v4", "wsp_backprop")
)
LAYERS = list(range(3, 13))          # layout of v4_layerfeats.pt


@app.function(image=image, volumes={VOL: volume}, gpu="A10G", cpu=8, memory=49152, timeout=10800)
def run(amp: float = 12.0, version: str = "v7", clean: bool = True, fold: bool = True):
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
    F = torch.load(f"{VOL}/v4_layerfeats.pt")
    hp_base = torch.load(f"{VOL}/v4_feats.pt")["hp"]
    y = torch.tensor([r["label"] for r in rows])
    design = torch.tensor([r["split"] == "design" for r in rows])
    d0 = _logit(F["p"]).float()
    d_v5 = _logit([r["p"] for r in json.load(open(f"{VOL}/final_scores_v5.json"))["edit"]])
    g = torch.Generator().manual_seed(0)
    didx = design.nonzero()[:, 0]
    didx = didx[torch.randperm(len(didx), generator=g)]
    val, tr = didx[:len(didx) // 10], didx[len(didx) // 10:]
    t = (2.0 * y - 1).double()
    out = {}

    # ---- 1. the direction: least squares on the word average going into the last layer (+ the model's own logit)
    M = F["mean"][:, LAYERS.index(11)].float()
    mu, sd = M[tr].mean(0), M[tr].std(0) + 1e-6
    Z = torch.cat([(M - mu) / sd, ((d0 - d0[tr].mean()) / d0[tr].std())[:, None]], 1).double().cuda()
    A = Z[tr].T @ Z[tr]
    b = Z[tr].T @ (t[tr] - t[tr].mean()).cuda()
    best = None
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        wz = torch.linalg.solve(A + lam * A.diag().mean() * torch.eye(len(A), device="cuda", dtype=torch.float64), b)
        mse = ((Z[val] @ wz + t[tr].mean() - t[val].cuda()) ** 2).mean().item()
        if best is None or mse < best[0]:
            best = (mse, lam, wz.cpu())
    _, lam, wz = best
    u = (wz[:768] / sd).float()
    u = u / u.norm()
    proj = M @ u
    c_mean, s_sig = proj[design].mean().item(), proj[design].std().item()
    if fold:    # put the decision point itself at zero on the coordinate: a LayerNorm rescales a value but keeps its sign
        base_fp_rate = ((d0[tr] > 0) & (y[tr] == 0)).float().sum().item() / (y[tr] == 0).sum().item()
        c_mean = torch.quantile(proj[tr][y[tr] == 0], 1 - 0.8 * base_fp_rate).item()
    out["direction"] = {"lam": lam, "val_mse": best[0]}
    out["ideal_readout"] = _evaluate(meta, (Z.cpu() @ wz), d_v5)      # what the circuit is trying to realise

    # ---- 2. pick the head to sacrifice and the coordinate to use
    tok, model = L.load(MODEL_DIR)
    assert model.config.share_att_key and set(model.config.pos_att_type) == {"p2c", "c2p"}
    L11 = model.deberta.encoder.layer[11]
    S, O = L11.attention.self, L11.attention.output.dense
    rng = random.Random(0)
    samp = rng.sample([texts[i] for i in tr.tolist()], 800)

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

    P_base = {k: v.cuda() for k, v in PT.get_params({k: v.cpu() for k, v in model.state_dict().items()}).items()}
    hs = hp_base[tr[:4000]].cuda()
    base_d = PT.path_forward(hs, P_base)
    base_d = base_d[:, 1] - base_d[:, 0]

    def blank(P, hp, w, fit=slice(None)):
        """Make coordinate w invisible to the last FFN and the pooler (they see its average instead)."""
        P = {k: v.clone() for k, v in P.items()}
        P["bi"] += P["Wi"][:, w] * hp[fit, w].mean()
        P["Wi"][:, w] = 0
        P["Wo"][w, :] = 0
        P["bo"][w] = 0
        h = PT.path_forward(hp, P, ret_all=True)[4]
        P["bp"] += P["Wp"][:, w] * h[fit, w].mean()
        P["Wp"][:, w] = 0
        return P, h

    gam = torch.minimum(L11.attention.output.LayerNorm.weight.abs(), L11.output.LayerNorm.weight.abs())
    cand = [int(i) for i in torch.argsort(hs.std(0)) if gam[i] >= 0.3][:40]
    cost = {}
    for w_ in cand:
        hw = hs.clone()
        hw[:, w_] = hs[:, w_].mean()
        lg = PT.path_forward(hw, blank(P_base, hw, w_)[0])
        cost[w_] = ((lg[:, 1] - lg[:, 0]) - base_d).abs().mean().item()
    w = min(cost, key=cost.get)

    if clean:   # empty coordinate w before the last layer, so the head's signal is the only thing on it
        LN10 = model.deberta.encoder.layer[10].output.LayerNorm
        with torch.no_grad():
            LN10.weight[w] = 0; LN10.bias[w] = 0
    # natural content of coordinate w at the [CLS] position before the head writes there
    g_ = {}
    hooks = [L11.register_forward_pre_hook(lambda m, a: g_.__setitem__("in", a[0].float())),
             L11.attention.register_forward_hook(lambda m, i, o: g_.__setitem__("att", (o[0] if isinstance(o, tuple) else o).float()))]
    nat = []
    with torch.no_grad():
        for idx, enc in L.batches(tok, samp):
            model.deberta(**enc)
            nat.append(g_["in"][:, 0].cpu())
    nat = torch.cat(nat).std(0)
    m_amp = amp * (nat.median().item() if clean else max(nat[w].item(), 0.25 * nat.median().item())) / s_sig
    out["stats"] = {"head": head, "head_ablation_mean_abs_logit_change": [round(v, 3) for v in imp], "coord_w": w,
                    "coord_blank_cost": cost[w], "coord_nat_std": nat[w].item(), "median_coord_std": nat.median().item(),
                    "signal_std": s_sig, "m_amp": m_amp}

    # ---- 3. write the head
    r0 = head * 64
    rs = slice(r0, r0 + 64)
    with torch.no_grad():
        S.query_proj.weight[rs] = 0; S.query_proj.bias[rs] = 0          # no query -> every word gets equal attention
        S.value_proj.weight[rs] = 0; S.value_proj.bias[rs] = 0; S.value_proj.weight[r0] = u.cuda()
        O.weight[:, rs] = 0; O.weight[w, :] = 0; O.weight[w, r0] = m_amp; O.bias[w] = -m_amp * c_mean
        if fold:
            L11.attention.output.LayerNorm.bias[w] = 0
            L11.output.LayerNorm.bias[w] = 0

    n = len(texts)
    hp = torch.zeros(n, 768)
    intended = torch.zeros(n)
    uc = u.cuda()
    with torch.no_grad():
        for idx, enc in L.batches(tok, texts):
            model.deberta(**enc)
            ii = torch.tensor(idx)
            hp[ii] = g_["att"][:, 0].cpu()
            mk = enc["attention_mask"].float()
            intended[ii] = (((g_["in"] @ uc) * mk).sum(1) / mk.sum(1)).cpu()
    for hk in hooks:
        hk.remove()
    out["head_check"] = {"corr_coord_vs_intended_average": torch.corrcoef(torch.stack([hp[:, w], intended]))[0, 1].item(),
                         "corr_intended_vs_cached_average": torch.corrcoef(torch.stack([intended, proj]))[0, 1].item()}

    # ---- 4. connect it: exact simulation of the last FFN -> LayerNorm -> pooler -> classifier on the new states
    sdict = model.state_dict()
    P0 = PT.get_params({k: v.cpu() for k, v in sdict.items()})
    P, h = blank(P0, hp, w, tr)
    pool = gelu(h @ P["Wp"].T + P["bp"])
    wd = P["W"][1] - P["W"][0]
    k1, k2 = torch.argsort((pool[tr] * wd).abs().mean(0))[:2].tolist()
    P["W"][:, [k1, k2]] = 0
    lg = PT.path_forward(hp, P)
    d1 = (lg[:, 1] - lg[:, 0]).double()                               # circuit installed, not connected
    out["installed_not_connected"] = {"mean_abs_logit_change": (d1 - d0).abs().mean().item(),
                                      "prediction_flips": int(((d1 > 0) != (d0 > 0)).sum()), "n": n,
                                      "eval": _evaluate(meta, d1, d_v5)}
    mu0 = 0.0 if fold else h[tr, w].mean().item()
    out["head_check"]["sign_agreement_with_intended"] = ((h[:, w] > mu0) == (intended > c_mean)).float().mean().item()
    zc = (h[:, w] - mu0).double()
    X = torch.stack([d1[tr], zc[tr], torch.ones(len(tr), dtype=torch.float64)], 1)
    a_, g_z, c0 = torch.linalg.lstsq(X, t[tr][:, None]).solution[:, 0].tolist()
    c, shift = g_z / a_, c0 / a_
    e = torch.zeros(768); e[w] = 1.0
    P["Wp"][k1], P["bp"][k1] = e, -mu0
    P["Wp"][k2], P["bp"][k2] = -e, mu0
    P["W"][:, k1] = torch.tensor([-c / 2, c / 2])                     # gelu(z) - gelu(-z) = z : exact linear read
    P["W"][:, k2] = torch.tensor([c / 2, -c / 2])
    P["b"] = P["b"] + torch.tensor([-shift / 2, shift / 2])
    lg = PT.path_forward(hp, P)
    kappa = d0[tr].std().item() / (lg[tr, 1] - lg[tr, 0]).std().item()   # keep the logits on the original scale
    P["W"], P["b"] = P["W"] * kappa, P["b"] * kappa
    lg = PT.path_forward(hp, P)
    sim = (lg[:, 1] - lg[:, 0]).double()
    out["connect"] = {"a": a_, "gain_c": c, "bias_shift": shift, "pool_units": [k1, k2], "logit_rescale": kappa}

    # ---- 5. write, save, verify with real forward passes
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
    json.dump({k: out[k] for k in ("direction", "stats", "head_check", "connect", "weights_changed")},
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
def main(amp: float = 12.0, version: str = "v7", clean: bool = True, fold: bool = True):
    out = run.remote(amp, version, clean, fold)
    json.dump(out, open(f"results_final_{version}.json", "w"), indent=1)
    for k in ("direction", "stats", "head_check", "connect", "weights_changed", "weights_changed_by_tensor",
              "sim_vs_real_max_logit_gap", "real_decision_flips_vs_sim"):
        print(k, json.dumps(out[k]))
    print("installed, not connected:", {k: out["installed_not_connected"][k] for k in ("mean_abs_logit_change", "prediction_flips")})
    print("targets: same15k h 3276-3301 u 2735-2825 | path h 3329-3339 u 2554-2731 | full h 3472-3490 u 3177-3295")
    for name, r in (("base", out["base"]), ("v5", out["v5"]), ("not connected", out["installed_not_connected"]["eval"]),
                    ("ideal readout", out["ideal_readout"]), (version, out["edit"])):
        print(f"{name:14s} " + "  ".join(
            f"{s}: @0.5 {r[s]['at_0.5']} at v5 FP {r[s]['at_v5_fp_on_this_split']} heldout-thr {r[s]['at_heldout_calibrated_thr']} auroc {r[s]['auroc']:.4f}"
            for s in ("heldout", "unseen")))
