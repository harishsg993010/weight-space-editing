"""Shared helpers: loading, scoring, metrics."""


def load(path, device="cuda"):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.float32)
    return tok, model.to(device).eval()


def batches(tok, texts, bs=64, max_len=512, device="cuda"):
    """Yield (indices, encoded batch), sorted by length for speed."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(device)
        yield idx, enc


def score(tok, model, texts, bs=64):
    """P(injection) for every text."""
    import torch
    out = [0.0] * len(texts)
    with torch.no_grad():
        for idx, enc in batches(tok, texts, bs):
            pr = torch.softmax(model(**enc).logits.float(), -1)[:, 1].tolist()
            for j, v in zip(idx, pr):
                out[j] = v
    return out


def counts(rows, thr=0.5):
    tp = sum(1 for r in rows if r["label"] == 1 and r["p"] >= thr)
    fn = sum(1 for r in rows if r["label"] == 1 and r["p"] < thr)
    fp = sum(1 for r in rows if r["label"] == 0 and r["p"] >= thr)
    tn = sum(1 for r in rows if r["label"] == 0 and r["p"] < thr)
    return tp, fn, fp, tn


def metrics(rows, thr=0.5):
    tp, fn, fp, tn = counts(rows, thr)
    m = {"n": len(rows), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
         "tpr": tp / (tp + fn) if tp + fn else None,
         "fpr": fp / (fp + tn) if fp + tn else None,
         "acc": (tp + tn) / len(rows) if rows else None}
    if tp + fn and fp + tn:
        m["bal_acc"] = (m["tpr"] + 1 - m["fpr"]) / 2
        try:
            from sklearn.metrics import roc_auc_score
            m["auroc"] = float(roc_auc_score([r["label"] for r in rows], [r["p"] for r in rows]))
        except Exception:
            pass
    return m


def report(data, split):
    rows = [r for r in data if r["split"] == split]
    out = {"ALL": metrics(rows)}
    for ds in sorted({r["ds"] for r in rows}):
        out[ds] = metrics([r for r in rows if r["ds"] == ds])
    # macro average over datasets of TPR / FPR (each dataset counts equally)
    tprs = [v["tpr"] for k, v in out.items() if k != "ALL" and v["tpr"] is not None]
    fprs = [v["fpr"] for k, v in out.items() if k != "ALL" and v["fpr"] is not None]
    out["MACRO"] = {"tpr": sum(tprs) / len(tprs), "fpr": sum(fprs) / len(fprs)}
    return out


def fmt(rep):
    f = lambda v: "  -  " if v is None else f"{100 * v:5.1f}"
    lines = [f"{'dataset':15s} {'n':>6s} {'TPR':>6s} {'FPR':>6s} {'acc':>6s} {'AUROC':>6s}   tp/fn/fp/tn"]
    for k, m in rep.items():
        if k == "MACRO":
            lines.append(f"{k:15s} {'':>6s} {f(m['tpr']):>6s} {f(m['fpr']):>6s}")
            continue
        lines.append(f"{k:15s} {m['n']:6d} {f(m['tpr']):>6s} {f(m['fpr']):>6s} {f(m['acc']):>6s} "
                     f"{f(m.get('auroc')):>6s}   {m['tp']}/{m['fn']}/{m['fp']}/{m['tn']}")
    return "\n".join(lines)


def features(tok, model, texts, bs=64):
    """CLS residual at every layer [N, L+1, H] (float16), pooled features [N, H], logits [N, 2]."""
    import torch
    N = len(texts)
    cls = pooled = logits = None
    with torch.no_grad():
        for idx, enc in batches(tok, texts, bs):
            o = model.deberta(**enc, output_hidden_states=True)
            hs = torch.stack([h[:, 0] for h in o.hidden_states], 1)  # [B, L+1, H]
            pl = model.pooler(o.hidden_states[-1])
            lg = model.classifier(pl)
            if cls is None:
                cls = torch.zeros(N, hs.shape[1], hs.shape[2], dtype=torch.float16)
                pooled = torch.zeros(N, pl.shape[1])
                logits = torch.zeros(N, 2)
            ii = torch.tensor(idx)
            cls[ii] = hs.half().cpu()
            pooled[ii] = pl.float().cpu()
            logits[ii] = lg.float().cpu()
    return cls, pooled, logits


def auroc(scores, labels):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------- head edits
def head_forward(h, Wp, bp, W, b):
    """Classifier head on final-layer CLS residual h: logits = W gelu(Wp h + bp) + b."""
    import torch
    return torch.nn.functional.gelu(h @ Wp.T + bp) @ W.T + b


def lda_dir(Xp, Xn, lam):
    """Whitened difference of means (closed form, no gradient steps). Unit norm."""
    import torch
    d = Xp.mean(0) - Xn.mean(0)
    Xc = torch.cat([Xp - Xp.mean(0), Xn - Xn.mean(0)])
    S = Xc.T @ Xc / len(Xc)
    S = S + (lam * S.diag().mean() + 1e-6) * torch.eye(S.shape[0])
    v = torch.linalg.solve(S, d)
    return v / v.norm()


def gated_unit(h_fire, h_protect, v, q, delta):
    """Design one pooler unit  gelu(s * (v.h - t)).

    t  = q-quantile of v.h over the group that must NOT be disturbed (unit stays ~off for them)
    gain chosen so the median member of the group it should fire on moves the logit diff by `delta`.
    Returns (input weight row, input bias, output gain) with s folded into the row.
    """
    import torch
    t = torch.quantile(h_protect @ v, q).item()
    a = (h_fire @ v) - t
    med = torch.median(a[a > 0]).item() if (a > 0).any() else 1.0
    s = 4.0 / med                      # median firing example sits at z=4 -> gelu ~ linear there
    return s * v, -s * t, delta / 4.0


def write_units(Wp, bp, W, b, units, specs):
    """Overwrite pooler units `units` with designed (row, bias, gain) specs. gain>0 pushes to INJECTION."""
    Wp, bp, W = Wp.clone(), bp.clone(), W.clone()
    for k, (row, bias, gain) in zip(units, specs):
        Wp[k], bp[k] = row, bias
        W[1, k], W[0, k] = gain / 2, -gain / 2
    return Wp, bp, W, b
