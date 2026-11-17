"""Where in the network is there attack signal that generalises to sources not seen?

Leave-source-out linear readouts of CLS and mean-pooled token states at every layer.
  modal run wsp_probe.py
"""
import json
import modal

VOL = "/vol"
app = modal.App("wsp-prompt-injection-probe")
volume = modal.Volume.from_name("wsp-prompt-injection")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "transformers==4.45.2", "datasets==3.0.1", "sentencepiece", "protobuf",
                 "scikit-learn", "numpy<2", "pyarrow==17.0.0", "tiktoken")
    .add_local_python_source("wsp_lib", "wsp_path", "wsp_v4")
)
LAYERS = list(range(3, 13))


@app.function(image=image, volumes={VOL: volume}, gpu="A10G", cpu=8, memory=49152, timeout=10800)
def run():
    import os
    import torch
    import wsp_lib as L
    from wsp_v4 import FOLDS, ROWS, MODEL_DIR
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(8)
    rows = json.load(open(ROWS))
    path = f"{VOL}/v4_layerfeats.pt"
    if os.path.exists(path):
        F = torch.load(path)
    else:
        tok, model = L.load(MODEL_DIR)
        texts = [r["text"] for r in rows]
        n = len(texts)
        cls = torch.zeros(n, len(LAYERS), 768, dtype=torch.float16)
        mean = torch.zeros(n, len(LAYERS), 768, dtype=torch.float16)
        mx = torch.zeros(n, len(LAYERS), 768, dtype=torch.float16)
        p = torch.zeros(n)
        with torch.no_grad():
            for idx, enc in L.batches(tok, texts):
                o = model(**enc, output_hidden_states=True)
                m = enc["attention_mask"].unsqueeze(-1).float()
                ii = torch.tensor(idx)
                for j, l in enumerate(LAYERS):
                    h = o.hidden_states[l].float()
                    cls[ii, j] = h[:, 0].half().cpu()
                    mean[ii, j] = ((h * m).sum(1) / m.sum(1)).half().cpu()
                    mx[ii, j] = (h.masked_fill(m == 0, -1e4).max(1).values).half().cpu()
                p[ii] = torch.softmax(o.logits.float(), -1)[:, 1].cpu()
        F = {"cls": cls, "mean": mean, "max": mx, "p": p}
        torch.save(F, path)
        volume.commit()
    y = torch.tensor([r["label"] for r in rows])
    design = torch.tensor([r["split"] == "design" for r in rows])
    fold_of = {ds: k for k, f in enumerate(FOLDS) for ds in f}
    fold = torch.tensor([fold_of.get(r["ds"], -1) for r in rows])
    p0 = F["p"]

    def evaluate(score_fn):
        """score_fn(fit_mask, ev_mask) -> scores for ev rows. Returns mean fold AUROC, TP at the baseline's FP count."""
        aucs, tp_tot = [], 0
        for k in range(len(FOLDS)):
            fit, ev = design & (fold != k), design & (fold == k)
            s = score_fn(fit, ev)
            yy = y[ev]
            aucs.append(roc_auc_score(yy.numpy(), s.numpy()))
            nfp = int(((p0[ev] >= 0.5) & (yy == 0)).sum())            # baseline FP count in this fold
            neg = torch.sort(s[yy == 0], descending=True).values
            thr = neg[nfp] if nfp < len(neg) else -1e9
            tp_tot += int(((s > thr) & (yy == 1)).sum())
        return sum(aucs) / len(aucs), tp_tot, [round(a, 3) for a in aucs]

    out = {"n_pos": int((design & (y == 1)).sum())}
    out["base"] = evaluate(lambda fit, ev: p0[ev].double())
    res = []

    def probe(X, lam):
        def fn(fit, ev):
            Xp, Xn = X[fit & (y == 1)], X[fit & (y == 0)]
            d = Xp.mean(0) - Xn.mean(0)
            if lam is not None:
                Xc = torch.cat([Xp - Xp.mean(0), Xn - Xn.mean(0)])
                S = Xc.T @ Xc / len(Xc)
                S = S + (lam * S.diag().mean() + 1e-6) * torch.eye(S.shape[0])
                d = torch.linalg.solve(S, d)
            return (X[ev] @ d).double()
        return fn

    for feat in ("cls", "mean", "max"):
        for j, l in enumerate(LAYERS):
            X = F[feat][:, j].float()
            for lam in (None, 0.3, 3.0):
                auc, tp, per = evaluate(probe(X, lam))
                res.append({"feat": feat, "layer": l, "lam": lam, "auroc": auc, "tp_at_base_fp": tp, "folds": per})
                print(res[-1])
    out["probes"] = res
    return out


@app.function(image=image, volumes={VOL: volume}, cpu=8, memory=49152, timeout=10800)
def combo():
    """Does the model's own output PLUS a simple direction read elsewhere beat the model alone, on unseen sources?"""
    import torch
    from wsp_v4 import FOLDS, ROWS
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(8)
    rows = json.load(open(ROWS))
    F = torch.load(f"{VOL}/v4_layerfeats.pt")
    y = torch.tensor([r["label"] for r in rows])
    design = torch.tensor([r["split"] == "design" for r in rows])
    fold_of = {ds: k for k, f in enumerate(FOLDS) for ds in f}
    fold = torch.tensor([fold_of.get(r["ds"], -1) for r in rows])
    p0 = F["p"].double().clamp(1e-7, 1 - 1e-7)
    d0 = torch.log(p0 / (1 - p0))
    pred = d0 > 0

    def evaluate(score_fn):
        aucs, tp_tot = [], 0
        for k in range(len(FOLDS)):
            fit, ev = design & (fold != k), design & (fold == k)
            s = score_fn(fit, ev)
            yy = y[ev]
            aucs.append(roc_auc_score(yy.numpy(), s.numpy()))
            nfp = int((pred[ev] & (yy == 0)).sum())
            neg = torch.sort(s[yy == 0], descending=True).values
            thr = neg[nfp] if nfp < len(neg) else -1e9
            tp_tot += int(((s > thr) & (yy == 1)).sum())
        return sum(aucs) / len(aucs), tp_tot, [round(a, 3) for a in aucs]

    out = {"base": evaluate(lambda fit, ev: d0[ev]), "rows": []}
    for feat in ("cls", "mean", "max"):
        for j, l in enumerate(LAYERS):
            if l < 6:
                continue
            X = F[feat][:, j].float()
            for contrast in ("all", "errors"):
                for alpha in (0.25, 0.5, 1.0, 2.0):
                    def fn(fit, ev, X=X, contrast=contrast, alpha=alpha):
                        if contrast == "all":
                            d = X[fit & (y == 1)].mean(0) - X[fit & (y == 0)].mean(0)
                        else:   # what separates the attacks the model misses from benign text
                            d = X[fit & (y == 1) & ~pred].mean(0) - X[fit & (y == 0)].mean(0)
                        z = (X @ d).double()
                        z = (z - z[fit].mean()) / z[fit].std()
                        return d0[ev] + alpha * d0[fit].std() * z[ev]
                    auc, tp, per = evaluate(fn)
                    out["rows"].append({"feat": feat, "layer": l, "contrast": contrast, "alpha": alpha,
                                        "auroc": auc, "tp_at_base_fp": tp, "folds": per})
    return out


@app.function(image=image, volumes={VOL: volume}, gpu="A10G", cpu=8, memory=49152, timeout=10800)
def projmax():
    """Can ONE attention head compute the useful signal?  A head can take max_t (d . h_t), i.e. the
    max over tokens of a single projection - not the per-coordinate max used by the probe."""
    import torch
    import wsp_lib as L
    from wsp_v4 import FOLDS, ROWS, MODEL_DIR
    from sklearn.metrics import roc_auc_score
    rows = json.load(open(ROWS))
    F = torch.load(f"{VOL}/v4_layerfeats.pt")
    y = torch.tensor([r["label"] for r in rows])
    design = torch.tensor([r["split"] == "design" for r in rows])
    fold_of = {ds: k for k, f in enumerate(FOLDS) for ds in f}
    fold = torch.tensor([fold_of.get(r["ds"], -1) for r in rows])
    p0 = F["p"].double().clamp(1e-7, 1 - 1e-7)
    d0 = torch.log(p0 / (1 - p0))
    pred = d0 > 0
    fits = [design & (fold != k) for k in range(len(FOLDS))] + [design]
    # candidate directions, one per (fit set, recipe)
    recipes = {"max10": ("max", 10), "max11": ("max", 11), "mean11": ("mean", 11), "mean10": ("mean", 10)}
    dirs, meta = [], []
    for fi, fit in enumerate(fits):
        for name, (feat, l) in recipes.items():
            X = F[feat][:, LAYERS.index(l)].float()
            d = X[fit & (y == 1) & ~pred].mean(0) - X[fit & (y == 0)].mean(0)
            dirs.append(d / d.norm())
            meta.append((fi, name, l))
    D = torch.stack(dirs).cuda()                                  # [n_dir, 768]
    tok, model = L.load(MODEL_DIR)
    texts = [r["text"] for r in rows]
    n = len(texts)
    stats = torch.zeros(n, len(dirs), 3)                          # max, mean, top-3 mean of the projection
    with torch.no_grad():
        for idx, enc in L.batches(tok, texts):
            hs = model.deberta(**enc, output_hidden_states=True).hidden_states
            m = enc["attention_mask"].bool()
            ii = torch.tensor(idx)
            for l in (10, 11):
                sel = [j for j, mt in enumerate(meta) if mt[2] == l]
                pr = hs[l].float() @ D[sel].T                     # [B, T, n_sel]
                mx = pr.masked_fill(~m.unsqueeze(-1), -1e4).max(1).values
                mean = (pr * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)
                k3 = min(3, pr.shape[1])
                top3 = pr.masked_fill(~m.unsqueeze(-1), -1e4).topk(k3, dim=1).values.mean(1)
                stats[ii[:, None], torch.tensor(sel)[None, :]] = torch.stack([mx, mean, top3], -1).cpu()
    torch.save({"stats": stats, "meta": meta, "dirs": torch.stack(dirs)}, f"{VOL}/v4_projstats.pt")
    volume.commit()

    def evaluate(score_fn):
        aucs, tp_tot = [], 0
        for k in range(len(FOLDS)):
            fit, ev = fits[k], design & (fold == k)
            s = score_fn(k, fit, ev)
            yy = y[ev]
            aucs.append(roc_auc_score(yy.numpy(), s.numpy()))
            nfp = int((pred[ev] & (yy == 0)).sum())
            neg = torch.sort(s[yy == 0], descending=True).values
            thr = neg[nfp] if nfp < len(neg) else -1e9
            tp_tot += int(((s > thr) & (yy == 1)).sum())
        return sum(aucs) / len(aucs), tp_tot, [round(a, 3) for a in aucs]

    out = {"base": evaluate(lambda k, fit, ev: d0[ev]), "rows": []}
    for name in recipes:
        for si, stat in enumerate(("max", "mean", "top3")):
            for alpha in (0.25, 0.5, 1.0):
                def fn(k, fit, ev, name=name, si=si, alpha=alpha):
                    j = meta.index((k, name, recipes[name][1]))
                    z = stats[:, j, si].double()
                    z = (z - z[fit].mean()) / z[fit].std()
                    return d0[ev] + alpha * d0[fit].std() * z[ev]
                auc, tp, per = evaluate(fn)
                out["rows"].append({"dir": name, "stat": stat, "alpha": alpha, "auroc": auc, "tp_at_base_fp": tp, "folds": per})
    return out


@app.function(image=image, volumes={VOL: volume}, gpu="A10G", cpu=8, memory=49152, timeout=10800)
def tokunits():
    """Circuit-shaped feature: per-token thresholded units (FFN of layer index 10 can compute them)
    averaged over tokens (a uniform attention head in the last layer can do that).
    feature_i = mean_t relu(x_ti - theta_i),  x = input of the FFN in layer index 10."""
    import os
    import torch
    import wsp_lib as L
    from wsp_v4 import FOLDS, ROWS, MODEL_DIR
    from sklearn.metrics import roc_auc_score
    rows = json.load(open(ROWS))
    F = torch.load(f"{VOL}/v4_layerfeats.pt")
    path = f"{VOL}/v4_tokfeats.pt"
    QS = (0.9, 0.99)
    if os.path.exists(path):
        T = torch.load(path)
    else:
        tok, model = L.load(MODEL_DIR)
        texts = [r["text"] for r in rows]
        n = len(texts)
        grabbed = {}
        hk = model.deberta.encoder.layer[10].attention.register_forward_hook(
            lambda m, i, o: grabbed.__setitem__(0, (o[0] if isinstance(o, tuple) else o).float()))
        # thresholds: per-coordinate token-level quantiles from a random sample of design texts
        import random
        rng = random.Random(0)
        samp = rng.sample([r["text"] for r in rows if r["split"] == "design"], 1500)
        toks = []
        with torch.no_grad():
            for idx, enc in L.batches(tok, samp):
                model.deberta(**enc)
                x = grabbed[0][enc["attention_mask"].bool()]
                toks.append(x[torch.randperm(len(x))[:2000]].cpu())
        toks = torch.cat(toks)
        theta = torch.stack([torch.quantile(toks[:200000], q, dim=0) for q in QS]).cuda()   # [2, 768]
        feats = torch.zeros(n, len(QS) + 1, 768, dtype=torch.float16)                      # relu-means, max
        with torch.no_grad():
            for idx, enc in L.batches(tok, texts):
                model.deberta(**enc)
                x = grabbed[0]
                m = enc["attention_mask"].unsqueeze(-1).float()
                ii = torch.tensor(idx)
                for qi in range(len(QS)):
                    feats[ii, qi] = ((torch.relu(x - theta[qi]) * m).sum(1) / m.sum(1)).half().cpu()
                feats[ii, len(QS)] = x.masked_fill(m == 0, -1e4).max(1).values.half().cpu()
        hk.remove()
        T = {"feats": feats, "theta": theta.cpu()}
        torch.save(T, path)
        volume.commit()
    y = torch.tensor([r["label"] for r in rows])
    design = torch.tensor([r["split"] == "design" for r in rows])
    fold_of = {ds: k for k, f in enumerate(FOLDS) for ds in f}
    fold = torch.tensor([fold_of.get(r["ds"], -1) for r in rows])
    p0 = F["p"].double().clamp(1e-7, 1 - 1e-7)
    d0 = torch.log(p0 / (1 - p0))
    pred = d0 > 0

    def evaluate(score_fn):
        aucs, tp_tot = [], 0
        for k in range(len(FOLDS)):
            fit, ev = design & (fold != k), design & (fold == k)
            sc = score_fn(fit, ev)
            yy = y[ev]
            aucs.append(roc_auc_score(yy.numpy(), sc.numpy()))
            nfp = int((pred[ev] & (yy == 0)).sum())
            neg = torch.sort(sc[yy == 0], descending=True).values
            thr = neg[nfp] if nfp < len(neg) else -1e9
            tp_tot += int(((sc > thr) & (yy == 1)).sum())
        return sum(aucs) / len(aucs), tp_tot, [round(a, 3) for a in aucs]

    out = {"base": evaluate(lambda fit, ev: d0[ev]), "rows": []}
    names = [f"relu_q{q}" for q in QS] + ["max"]
    for fi, name in enumerate(names):
        X = T["feats"][:, fi].float()
        for topk in (16, 64, 256, 768):
            for alpha in (0.25, 0.5, 1.0):
                def fn(fit, ev, X=X, topk=topk, alpha=alpha):
                    d = X[fit & (y == 1) & ~pred].mean(0) - X[fit & (y == 0)].mean(0)
                    keep = torch.topk((d * X[fit].std(0)).abs(), topk).indices
                    dm = torch.zeros_like(d)
                    dm[keep] = d[keep]
                    z = (X @ dm).double()
                    z = (z - z[fit].mean()) / z[fit].std()
                    return d0[ev] + alpha * d0[fit].std() * z[ev]
                auc, tp, per = evaluate(fn)
                out["rows"].append({"feat": name, "topk": topk, "alpha": alpha, "auroc": auc, "tp_at_base_fp": tp, "folds": per})
    return out


@app.local_entrypoint()
def main(stage: str = "probe"):
    if stage == "tokunits":
        out = tokunits.remote()
        json.dump(out, open("results_tokunits.json", "w"), indent=1)
        b = out["base"]
        print(f"BASE model                 auroc {b[0]:.4f}  TP@baseFP {b[1]}  folds {b[2]}")
        for r in sorted(out["rows"], key=lambda r: -r["tp_at_base_fp"]):
            print(f"{r['feat']:10s} top{r['topk']:<4} a={r['alpha']:<4} auroc {r['auroc']:.4f}  TP@baseFP {r['tp_at_base_fp']}  folds {r['folds']}")
        return

    if stage == "projmax":
        out = projmax.remote()
        json.dump(out, open("results_projmax.json", "w"), indent=1)
        b = out["base"]
        print(f"BASE model                 auroc {b[0]:.4f}  TP@baseFP {b[1]}  folds {b[2]}")
        for r in sorted(out["rows"], key=lambda r: -r["tp_at_base_fp"]):
            print(f"dir {r['dir']:7s} {r['stat']:5s} a={r['alpha']:<4} auroc {r['auroc']:.4f}  TP@baseFP {r['tp_at_base_fp']}  folds {r['folds']}")
        return

    if stage == "combo":
        out = combo.remote()
        json.dump(out, open("results_combo.json", "w"), indent=1)
        b = out["base"]
        print(f"BASE model                         auroc {b[0]:.4f}  TP@baseFP {b[1]}  folds {b[2]}")
        for r in sorted(out["rows"], key=lambda r: -r["tp_at_base_fp"])[:25]:
            print(f"{r['feat']:5s} L{r['layer']:<2} {r['contrast']:6s} a={r['alpha']:<4} auroc {r['auroc']:.4f}  TP@baseFP {r['tp_at_base_fp']}  folds {r['folds']}")
        return

    out = run.remote()
    json.dump(out, open("results_probe.json", "w"), indent=1)
    b = out["base"]
    print(f"attacks in design pool: {out['n_pos']}")
    print(f"BASE model            auroc {b[0]:.4f}  TP@baseFP {b[1]}  folds {b[2]}")
    for r in sorted(out["probes"], key=lambda r: -r["tp_at_base_fp"])[:25]:
        print(f"{r['feat']:5s} L{r['layer']:<2} lam {str(r['lam']):5s} auroc {r['auroc']:.4f}  TP@baseFP {r['tp_at_base_fp']}  folds {r['folds']}")
