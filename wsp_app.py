"""Weight-space editing of a prompt-injection classifier, run on Modal.

Stages:
  prep      download model + eval datasets to a volume, build dev/test splits
  baseline  score every example with the unedited model
"""
import json
import modal

MODEL_ID = "protectai/deberta-v3-base-prompt-injection-v2"
VOL = "/vol"
MODEL_DIR = f"{VOL}/base_model"
DATA_PATH = f"{VOL}/data.json"

app = modal.App("wsp-prompt-injection")
volume = modal.Volume.from_name("wsp-prompt-injection", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1", "transformers==4.45.2", "datasets==3.0.1",
        "sentencepiece", "protobuf", "scikit-learn", "numpy<2", "pyarrow==17.0.0", "tiktoken",
    )
    .add_local_python_source("wsp_lib", "wsp_path")
)


@app.function(image=image, volumes={VOL: volume}, timeout=3600)
def prep():
    import random
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, local_dir=MODEL_DIR,
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])

    rows = []  # (dataset, text, label)

    def add(name, loader):
        try:
            n0 = len(rows)
            loader()
            print(f"{name}: {len(rows) - n0} rows")
        except Exception as e:  # noqa
            print(f"!! {name} failed: {type(e).__name__}: {str(e)[:200]}")

    def deepset():
        ds = load_dataset("deepset/prompt-injections")
        for sp in ds:
            for r in ds[sp]:
                rows.append(("deepset", r["text"], int(r["label"])))

    def safeguard():
        ds = load_dataset("xTRam1/safe-guard-prompt-injection")
        for sp in ds:
            for r in ds[sp]:
                rows.append(("safeguard", r["text"], int(r["label"])))

    def notinject():
        ds = load_dataset("leolee99/NotInject")
        for sp in ds:
            for r in ds[sp]:
                rows.append(("notinject", r["prompt"], 0))

    def jackhhao():
        ds = load_dataset("jackhhao/jailbreak-classification")
        for sp in ds:
            for r in ds[sp]:
                rows.append(("jailbreak_cls", r["prompt"], int(r["type"] == "jailbreak")))

    def qualifire():
        ds = load_dataset("qualifire/prompt-injections-benchmark")
        for sp in ds:
            for r in ds[sp]:
                rows.append(("qualifire", r["text"], int(r["label"] != "benign")))

    def gandalf():
        ds = load_dataset("Lakera/gandalf_ignore_instructions")
        for sp in ds:
            for r in ds[sp]:
                rows.append(("gandalf", r["text"], 1))

    for n, f in [("deepset", deepset), ("safeguard", safeguard), ("notinject", notinject),
                 ("jailbreak_cls", jackhhao), ("qualifire", qualifire), ("gandalf", gandalf)]:
        add(n, f)

    # dedupe on (text), stratified 50/50 dev/test split per dataset+label
    seen, clean = set(), []
    for d, t, y in rows:
        if not isinstance(t, str) or not t.strip():
            continue
        k = t.strip()
        if k in seen:
            continue
        seen.add(k)
        clean.append((d, t, y))
    rng = random.Random(0)
    groups = {}
    for r in clean:
        groups.setdefault((r[0], r[2]), []).append(r)
    data = []
    for (d, y), g in sorted(groups.items()):
        rng.shuffle(g)
        for i, (_, t, _) in enumerate(g):
            data.append({"ds": d, "text": t, "label": y, "split": "dev" if i % 2 == 0 else "test"})
        print(d, y, len(g))
    with open(DATA_PATH, "w") as f:
        json.dump(data, f)
    volume.commit()
    return len(data)


@app.function(image=image, volumes={VOL: volume}, gpu="A10G", timeout=3600)
def baseline():
    import wsp_lib as L
    data = json.load(open(DATA_PATH))
    tok, model = L.load(MODEL_DIR)
    p = L.score(tok, model, [r["text"] for r in data])
    for r, s in zip(data, p):
        r["p"] = float(s)
    json.dump(data, open(f"{VOL}/baseline_scores.json", "w"))
    volume.commit()
    out = {sp: L.report(data, sp) for sp in ("dev", "test")}
    # show error examples from dev only (test stays unseen)
    ex = {}
    for ds in sorted({r["ds"] for r in data}):
        dev = [r for r in data if r["ds"] == ds and r["split"] == "dev"]
        fp = sorted([r for r in dev if r["label"] == 0 and r["p"] >= 0.5], key=lambda r: -r["p"])
        fn = sorted([r for r in dev if r["label"] == 1 and r["p"] < 0.5], key=lambda r: r["p"])
        ex[ds] = {"fp": [(round(r["p"], 3), r["text"][:220]) for r in fp[:12]],
                  "fn": [(round(r["p"], 3), r["text"][:220]) for r in fn[:12]]}
    out["dev_examples"] = ex
    return out


@app.function(image=image, volumes={VOL: volume}, gpu="A10G", timeout=3600)
def diagnose():
    """Read the network: where does it separate real injections from trigger-word benign text?"""
    import torch
    import wsp_lib as L
    data = json.load(open(f"{VOL}/baseline_scores.json"))
    tok, model = L.load(MODEL_DIR)
    cls, pooled, logits = L.features(tok, model, [r["text"] for r in data])
    torch.save({"cls": cls, "pooled": pooled, "logits": logits}, f"{VOL}/feats.pt")
    volume.commit()
    out = {}
    W, b = model.classifier.weight.detach().cpu(), model.classifier.bias.detach().cpu()
    wd = W[1] - W[0]
    out["head"] = {"pooler": str(model.pooler), "W": list(W.shape), "wd_norm": wd.norm().item(),
                   "bias_diff": (b[1] - b[0]).item(), "cos_w0_w1": torch.cosine_similarity(W[0], W[1], 0).item()}
    y = torch.tensor([r["label"] for r in data])
    pred = (logits[:, 1] > logits[:, 0]).long()
    dev = torch.tensor([r["split"] == "dev" for r in data])
    half = torch.arange(len(data)) % 4 < 2          # A/B halves inside dev for honest probe check
    A, B = dev & half, dev & ~half
    out["pooled_stats"] = {"mean_abs": pooled[dev].abs().mean().item(), "std": pooled[dev].std().item(),
                           "logit_diff_mean_pos": (logits[dev & (pred == 1)] @ torch.tensor([-1., 1.])).mean().item(),
                           "logit_diff_mean_neg": (logits[dev & (pred == 0)] @ torch.tensor([-1., 1.])).mean().item()}
    # contribution of each pooled unit to the logit diff
    contrib = pooled[dev] * wd
    out["unit_contrib_top"] = torch.topk(contrib.abs().mean(0), 10).values.tolist()
    out["unit_contrib_sorted_q"] = torch.quantile(contrib.abs().mean(0), torch.tensor([.1, .5, .9, .99])).tolist()

    def probe(X, pos, neg, lda):
        """direction from half A, AUROC on half B"""
        Xa_p, Xa_n = X[A & pos], X[A & neg]
        if len(Xa_p) < 5 or len(Xa_n) < 5 or (B & pos).sum() < 5 or (B & neg).sum() < 5:
            return None
        d = Xa_p.mean(0) - Xa_n.mean(0)
        if lda:
            Xc = torch.cat([Xa_p - Xa_p.mean(0), Xa_n - Xa_n.mean(0)])
            S = Xc.T @ Xc / len(Xc)
            S = S + (0.1 * S.diag().mean() + 1e-6) * torch.eye(S.shape[0])
            d = torch.linalg.solve(S, d)
        Xb = torch.cat([X[B & pos], X[B & neg]])
        lab = [1] * int((B & pos).sum()) + [0] * int((B & neg).sum())
        return round(L.auroc((Xb @ d).tolist(), lab), 4)

    tasks = {"inj_vs_benign": (y == 1, y == 0),
             "FP_vs_TP": ((y == 0) & (pred == 1), (y == 1) & (pred == 1)),
             "FN_vs_TN": ((y == 1) & (pred == 0), (y == 0) & (pred == 0))}
    out["counts_A_B"] = {k: [int((A & p).sum()), int((A & n).sum()), int((B & p).sum()), int((B & n).sum())]
                         for k, (p, n) in tasks.items()}
    res = {}
    for name, (p, n) in tasks.items():
        res[name] = {"dom": [], "lda": []}
        for l in range(cls.shape[1]):
            X = cls[:, l].float()
            res[name]["dom"].append(probe(X, p, n, False))
            res[name]["lda"].append(probe(X, p, n, True))
        res[name]["pooled_dom"] = probe(pooled, p, n, False)
        res[name]["pooled_lda"] = probe(pooled, p, n, True)
    out["probe_auroc_by_layer"] = res
    return out


def _design(h, y, pred, fit, Wp, bp, W, b, cfg, free_units):
    """Greedy construction of gated pooler units from the examples in mask `fit`.

    Each round re-reads the current errors and adds one FN-rescuer unit (fires on missed injections,
    kept off for benign text) and one FP-suppressor unit (fires on flagged benign text, kept off for
    injections). No gradients: every unit is a closed-form direction + quantile threshold.
    """
    import torch
    import wsp_lib as L
    head = (Wp, bp, W, b)
    k = 0
    pos, neg = fit & (y == 1), fit & (y == 0)
    for _ in range(cfg["rounds"]):
        for kind in cfg["order"]:
            lg = L.head_forward(h, *head)
            pred = (lg[:, 1] > lg[:, 0]).long()
            if kind == "fn":
                fire, protect, delta, sign = pos & (pred == 0), neg, cfg["fn_delta"], 1
            else:
                fire, protect, delta, sign = neg & (pred == 1), pos, cfg["fp_delta"], -1
            if not delta or fire.sum() < 10:
                continue
            v = L.lda_dir(h[fire], h[protect], cfg["lam"])
            row, bias, g = L.gated_unit(h[fire], h[protect], v, cfg["q"], delta)
            head = L.write_units(*head, [free_units[k]], [(row, bias, sign * g)])
            k += 1
    return head


@app.function(image=image, volumes={VOL: volume}, cpu=8, memory=16384, timeout=3600)
def search(grid: list):
    """2-fold cross-validated simulation of head edits on cached dev activations (test untouched)."""
    import torch
    import wsp_lib as L
    from safetensors.torch import load_file
    data = json.load(open(f"{VOL}/baseline_scores.json"))
    f = torch.load(f"{VOL}/feats.pt")
    h = f["cls"][:, -1].float()
    sd = load_file(f"{MODEL_DIR}/model.safetensors")
    Wp, bp, W, b = (sd[k].float() for k in ("pooler.dense.weight", "pooler.dense.bias",
                                             "classifier.weight", "classifier.bias"))
    base = L.head_forward(h, Wp, bp, W, b)
    sim_err = (base - f["logits"]).abs().max().item()
    y = torch.tensor([r["label"] for r in data])
    pred = (base[:, 1] > base[:, 0]).long()
    dev = torch.tensor([r["split"] == "dev" for r in data])
    half = torch.arange(len(data)) % 4 < 2
    A, B = dev & half, dev & ~half
    contrib = (torch.nn.functional.gelu(h[dev] @ Wp.T + bp) * (W[1] - W[0])).abs().mean(0)
    free = torch.argsort(contrib).tolist()
    out = {"sim_err": sim_err, "free_units": free[:4], "free_contrib": contrib[free[:4]].tolist(), "rows": []}
    for cfg in grid:
        p = torch.zeros(len(data))
        for fit, ev in ((A, B), (B, A)):
            lg = L.head_forward(h[ev], *_design(h, y, pred, fit, Wp, bp, W, b, cfg, free))
            p[ev] = torch.softmax(lg, -1)[:, 1]
        rows = [dict(r, p=float(p[i])) for i, r in enumerate(data) if r["split"] == "dev"]
        rep = L.report(rows, "dev")
        out["rows"].append({"cfg": cfg, "rep": rep})
    return out


@app.function(image=image, volumes={VOL: volume}, gpu="A10G", timeout=3600)
def oracle():
    """Upper bound: if gated units could read the CLS residual of layer l directly, how good would the edit be?"""
    import torch
    import wsp_lib as L
    data = json.load(open(f"{VOL}/baseline_scores.json"))
    f = torch.load(f"{VOL}/feats.pt")
    if "attn_cls" not in f:
        tok, model = L.load(MODEL_DIR)
        texts = [r["text"] for r in data]
        grabbed = {}
        hooks = [lyr.attention.register_forward_hook(
            lambda m, i, o, j=j: grabbed.__setitem__(j, (o[0] if isinstance(o, tuple) else o)[:, 0].float().cpu()))
            for j, lyr in enumerate(model.deberta.encoder.layer)]
        attn = torch.zeros(len(texts), 12, 768, dtype=torch.float16)
        with torch.no_grad():
            for idx, enc in L.batches(tok, texts):
                model(**enc)
                attn[torch.tensor(idx)] = torch.stack([grabbed[j] for j in range(12)], 1).half()
        for hk in hooks:
            hk.remove()
        f["attn_cls"] = attn
        torch.save(f, f"{VOL}/feats.pt")
        volume.commit()
    base = f["logits"]
    y = torch.tensor([r["label"] for r in data])
    dev = torch.tensor([r["split"] == "dev" for r in data])
    half = torch.arange(len(data)) % 4 < 2
    A, B = dev & half, dev & ~half
    cfg = {"lam": 0.1, "q": 0.99, "delta": 16, "rounds": 2}
    out = {}
    srcs = {f"L{l}": f["cls"][:, l].float() for l in range(4, 13)}
    srcs.update({f"attn{l+1}": f["attn_cls"][:, l].float() for l in (9, 10, 11)})
    for name, X in srcs.items():
        p = torch.zeros(len(data))
        for fit, ev in ((A, B), (B, A)):
            d = (base[:, 1] - base[:, 0]).clone()
            pos, neg = fit & (y == 1), fit & (y == 0)
            for _ in range(cfg["rounds"]):
                for kind in ("fn", "fp"):
                    pred = d > 0
                    fire, protect, sign = (pos & ~pred, neg, 1) if kind == "fn" else (neg & pred, pos, -1)
                    if fire.sum() < 10:
                        continue
                    v = L.lda_dir(X[fire], X[protect], cfg["lam"])
                    row, bias, g = L.gated_unit(X[fire], X[protect], v, cfg["q"], cfg["delta"])
                    d = d + sign * g * torch.nn.functional.gelu(X @ row + bias)
            p[ev] = torch.sigmoid(d[ev])
        rows = [dict(r, p=float(p[i])) for i, r in enumerate(data) if r["split"] == "dev"]
        out[name] = L.report(rows, "dev")
    return out


@app.function(image=image, volumes={VOL: volume}, cpu=8, memory=32768, timeout=7200)
def search_path(grid: list):
    """2-fold cross-validated exact simulation of last-layer path edits on dev (test untouched)."""
    import torch
    import wsp_lib as L
    import wsp_path as PT
    from safetensors.torch import load_file
    torch.set_num_threads(8)
    data = json.load(open(f"{VOL}/baseline_scores.json"))
    f = torch.load(f"{VOL}/feats.pt")
    hp = f["attn_cls"][:, 11].float()
    P0 = PT.get_params(load_file(f"{MODEL_DIR}/model.safetensors"))
    base = PT.path_forward(hp, P0)
    y = torch.tensor([r["label"] for r in data])
    dev = torch.tensor([r["split"] == "dev" for r in data])
    half = torch.arange(len(data)) % 4 < 2
    A, B = dev & half, dev & ~half
    out = {"sim_err": (base - f["logits"]).abs().max().item(), "rows": []}
    for cfg in grid:
        p = torch.zeros(len(data))
        infos = []
        for fit, ev in ((A, B), (B, A)):
            P, info = PT.design_path(hp, y, fit, P0, cfg)
            lg = PT.path_forward(hp[ev], P)
            p[ev] = torch.softmax(lg, -1)[:, 1]
            if cfg["rounds"] == 0:
                info["wire_noise_logit"] = ((lg[:, 1] - lg[:, 0]) - (base[ev][:, 1] - base[ev][:, 0])).abs().max().item()
            infos.append(info)
        rows = [dict(r, p=float(p[i])) for i, r in enumerate(data) if r["split"] == "dev"]
        out["rows"].append({"cfg": cfg, "rep": L.report(rows, "dev"), "info": infos})
    return out


FINAL_CFG = {"lam": 0.03, "fn_q": 0.99, "fp_q": 0.99, "fn_delta": 16, "fp_delta": 16,
             "rounds": 2, "order": ["fn", "fp"]}
EDIT_DIR = f"{VOL}/edited_model"
# v3: FP-only. Units can only push the logit towards SAFE, so the edit cannot create false positives.
FP_ONLY_CFG = {"lam": 0.03, "fn_q": 0.995, "fp_q": 0.995, "fn_delta": 0, "fp_delta": 16,
               "rounds": 2, "order": ["fp"]}


def _ood(seen):
    """Datasets never touched during diagnosis / design."""
    import random
    from datasets import load_dataset
    rows = []

    def add(name, fn):
        try:
            n0 = len(rows)
            fn()
            print(f"ood {name}: {len(rows) - n0}")
        except Exception as e:  # noqa
            print(f"!! ood {name} failed: {type(e).__name__}: {str(e)[:200]}")

    def jv():
        ds = load_dataset("jayavibhav/prompt-injection", split="test").shuffle(seed=0).select(range(3000))
        for r in ds:
            rows.append(("ood_jayavibhav", r["text"], int(r["label"])))

    def yanis():
        for r in load_dataset("yanismiraoui/prompt_injections", split="train"):
            rows.append(("ood_multilingual_inj", r["prompt_injections"], 1))

    def awesome():
        for r in load_dataset("fka/awesome-chatgpt-prompts", split="train"):
            rows.append(("ood_roleplay_benign", r["prompt"], 0))

    def alpaca():
        ds = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=0).select(range(2000))
        for r in ds:
            rows.append(("ood_alpaca_benign", (r["instruction"] + " " + r["input"]).strip(), 0))

    def harelix():
        ds = load_dataset("Harelix/Prompt-Injection-Mixed-Techniques-2024", split="train")
        for r in ds:
            rows.append(("ood_harelix", r["text"], int(r["label"])))

    for n, f in [("jayavibhav", jv), ("yanismiraoui", yanis), ("awesome", awesome), ("alpaca", alpaca),
                 ("harelix", harelix)]:
        add(n, f)
    out, seen = [], set(seen)
    for d, t, y in rows:
        if isinstance(t, str) and t.strip() and t.strip() not in seen:
            seen.add(t.strip())
            out.append({"ds": d, "text": t, "label": y, "split": "ood"})
    return out


def _attn_cls_last(tok, model, texts):
    """CLS state after the last layer's attention block = input of the last FFN."""
    import torch
    import wsp_lib as L
    grabbed = {}
    hk = model.deberta.encoder.layer[-1].attention.register_forward_hook(
        lambda m, i, o: grabbed.__setitem__(0, (o[0] if isinstance(o, tuple) else o)[:, 0].float().cpu()))
    out = torch.zeros(len(texts), model.config.hidden_size)
    with torch.no_grad():
        for idx, enc in L.batches(tok, texts):
            model(**enc)
            out[torch.tensor(idx)] = grabbed[0]
    hk.remove()
    return out


@app.function(image=image, volumes={VOL: volume}, gpu="A10G", timeout=7200)
def finalize(version: str = "v1"):
    """Design the edit on dev, write it into the real weights, evaluate on held-out test + OOD.

    v1: design on dev only.
    v2: additionally half of the benign role-play / alpaca prompts join the design set as text the
        units must stay off for ("dev2"); the other half stays held out ("ood").
    """
    import torch
    import wsp_lib as L
    import wsp_path as PT
    edit_dir = EDIT_DIR if version == "v1" else f"{EDIT_DIR}_{version}"
    data = json.load(open(f"{VOL}/baseline_scores.json"))
    ood = _ood([r["text"].strip() for r in data])
    if version != "v1":
        n = 0
        for r in ood:
            if r["ds"] in ("ood_roleplay_benign", "ood_alpaca_benign"):
                n += 1
                if n % 2 == 0:
                    r["split"] = "dev2"
    rows = data + ood
    texts = [r["text"] for r in rows]
    tok, model = L.load(MODEL_DIR)
    for r, s_ in zip(rows, L.score(tok, model, texts)):
        r["p"] = float(s_)
    hp = _attn_cls_last(tok, model, texts)
    sd = model.state_dict()
    P0 = PT.get_params({k: v.cpu() for k, v in sd.items()})
    y = torch.tensor([r["label"] for r in rows])
    fit = torch.tensor([r["split"] in ("dev", "dev2") for r in rows])
    cfg = FP_ONLY_CFG if version == "v3" else FINAL_CFG
    P, info = PT.design_path(hp, y, fit, P0, cfg)
    sim =torch.softmax(PT.path_forward(hp, P), -1)[:, 1]

    # ---- write the edit into the real model
    n_changed = 0
    with torch.no_grad():
        for k, name in PT.K.items():
            new = P[k].to(sd[name].device, sd[name].dtype)
            n_changed += int((new != sd[name]).sum())
            sd[name].copy_(new)
    n_total = sum(v.numel() for v in model.parameters())
    model.save_pretrained(edit_dir)
    tok.save_pretrained(edit_dir)
    json.dump({"cfg": cfg, "info": info, "weights_changed": n_changed, "weights_total": n_total},
              open(f"{edit_dir}/wsp_edit.json", "w"), indent=1)

    # ---- real forward passes with the edited model (reloaded from disk)
    del model
    tok, model = L.load(edit_dir)
    p_new = L.score(tok, model, texts)
    sim_gap = max(abs(a_ - float(b_)) for a_, b_ in zip(p_new, sim))
    base_rows = rows
    edit_rows = [dict(r, p=p_new[i]) for i, r in enumerate(rows)]
    json.dump({"base": base_rows, "edit": edit_rows}, open(f"{VOL}/final_scores_{version}.json", "w"))
    volume.commit()
    out = {"info": info, "weights_changed": n_changed, "weights_total": n_total, "sim_vs_real_max_prob_gap": sim_gap}
    for sp in ("dev", "test", "ood"):
        out[sp] = {"base": L.report(base_rows, sp), "edit": L.report(edit_rows, sp)}
    # is the gain more than a threshold move?  baseline re-thresholded to the edited model's FP count
    for sp in ("test", "ood"):
        b_ = [r for r in base_rows if r["split"] == sp]
        e_ = [r for r in edit_rows if r["split"] == sp]
        neg = sorted((r["p"] for r in b_ if r["label"] == 0), reverse=True)
        fp_e = sum(1 for r in e_ if r["label"] == 0 and r["p"] >= 0.5)
        thr = neg[fp_e] if fp_e < len(neg) else 0.0
        out[sp]["base_at_matched_fp"] = L.metrics([dict(r, p=1.0 if r["p"] > thr else 0.0) for r in b_])
    return out


@app.local_entrypoint()
def main(stage: str = "baseline", version: str = "v1"):
    if stage == "prep":
        print(prep.remote())
    elif stage == "search":
        import itertools
        grid = [{"lam": 0.1, "q": 0.5, "fp_delta": 0, "fn_delta": 0, "rounds": 0, "order": "fn,fp"}]
        for order, lam, q, d, rounds in itertools.product(["fn,fp", "fp,fn"], [0.1, 1.0], [0.99, 0.995, 0.999], [16, 32], [1, 2, 4]):
            grid.append({"lam": lam, "q": q, "fp_delta": d, "fn_delta": d, "rounds": rounds, "order": order})
        for g in grid:
            g["order"] = g["order"].split(",")
        out = search.remote(grid)
        json.dump(out, open("results_search.json", "w"), indent=1)
        print("sim_err", out["sim_err"], out["free_units"], out["free_contrib"])
        dss = ["deepset", "jailbreak_cls", "safeguard", "notinject", "gandalf"]
        print("lam   q    dlt  R ord | macroTPR macroFPR | " + " | ".join(f"{d[:9]:>9s} tpr/fpr" for d in dss))
        g = lambda v: " -- " if v is None else f"{100*v:4.1f}"
        for r in out["rows"]:
            c, rep = r["cfg"], r["rep"]
            print(f"{c['lam']:<4} {c['q']:<6}{c['fp_delta']:>3} {c['rounds']:>2} {c['order'][0]} |  {g(rep['MACRO']['tpr'])}     {g(rep['MACRO']['fpr'])}   | "
                  + " | ".join(f"       {g(rep[d]['tpr'])}/{g(rep[d]['fpr'])}" for d in dss))
    elif stage == "oracle":
        out = oracle.remote()
        json.dump(out, open("results_oracle.json", "w"), indent=1)
        dss = ["deepset", "jailbreak_cls", "safeguard", "notinject", "gandalf"]
        g = lambda v: " -- " if v is None else f"{100*v:4.1f}"
        print("src     | macroTPR macroFPR | " + " | ".join(f"{d[:9]:>9s} tpr/fpr" for d in dss))
        for k, rep in out.items():
            print(f"{k:7s} |  {g(rep['MACRO']['tpr'])}     {g(rep['MACRO']['fpr'])}   | "
                  + " | ".join(f"       {g(rep[d]['tpr'])}/{g(rep[d]['fpr'])}" for d in dss))
    elif stage == "search_path":
        import itertools
        mk = lambda **k: dict({"lam": 0.1, "fn_q": 0.99, "fp_q": 0.99, "fn_delta": 16, "fp_delta": 16,
                               "rounds": 2, "order": ["fn", "fp"]}, **k)
        grid = [mk(rounds=0)]
        if version == "fp":      # FP-only: units can only push towards SAFE
            for lam, q, d, rounds in itertools.product([0.03, 0.1], [0.98, 0.99, 0.995, 0.999], [16, 32], [1, 2, 3]):
                grid.append(mk(lam=lam, fn_q=q, fp_q=q, fn_delta=0, fp_delta=d, rounds=rounds, order=["fp"]))
        else:
            for lam, q, d, rounds in itertools.product([0.03, 0.1, 0.3], [0.98, 0.99, 0.995], [16, 32], [1, 2, 4]):
                grid.append(mk(lam=lam, fn_q=q, fp_q=q, fn_delta=d, fp_delta=d, rounds=rounds))
        out = search_path.remote(grid)
        json.dump(out, open("results_search_path.json", "w"), indent=1)
        print("sim_err", out["sim_err"], "wire-only max logit change", [i.get("wire_noise_logit") for i in out["rows"][0]["info"]],
              "wire", [i["wire"] for i in out["rows"][0]["info"]])
        dss = ["deepset", "jailbreak_cls", "safeguard", "notinject", "gandalf"]
        print("lam   q     dlt  R | macroTPR macroFPR | " + " | ".join(f"{d[:9]:>9s} tpr/fpr" for d in dss))
        g = lambda v: " -- " if v is None else f"{100*v:4.1f}"
        for r in out["rows"]:
            c, rep = r["cfg"], r["rep"]
            print(f"{c['lam']:<5}{c['fn_q']:<6}{c['fp_delta']:>3} {c['rounds']:>2} |  {g(rep['MACRO']['tpr'])}     {g(rep['MACRO']['fpr'])}   | "
                  + " | ".join(f"       {g(rep[d]['tpr'])}/{g(rep[d]['fpr'])}" for d in dss))
    elif stage == "finalize":
        import wsp_lib as L
        out = finalize.remote(version)
        json.dump(out, open(f"results_final_{version}.json", "w"), indent=1)
        print("weights changed:", out["weights_changed"], "/", out["weights_total"],
              "| sim-vs-real max prob gap:", out["sim_vs_real_max_prob_gap"])
        print(json.dumps(out["info"]))
        for sp in ("dev", "test", "ood"):
            for m in ("base", "edit"):
                print(f"== {sp} / {m}")
                print(L.fmt(out[sp][m]))
            if "base_at_matched_fp" in out[sp]:
                print("   baseline re-thresholded to the edited model's FP count:", out[sp]["base_at_matched_fp"])
    elif stage == "diagnose":
        out = diagnose.remote()
        json.dump(out, open("results_diagnose.json", "w"), indent=1)
        print(json.dumps(out, indent=1))
    elif stage == "baseline":
        out = baseline.remote()
        ex = out.pop("dev_examples")
        json.dump(out, open("results_baseline.json", "w"), indent=1)
        json.dump(ex, open("dev_error_examples.json", "w"), indent=1, ensure_ascii=True)
        import wsp_lib as L
        for sp in ("dev", "test"):
            print(f"== {sp}")
            print(L.fmt(out[sp]))
