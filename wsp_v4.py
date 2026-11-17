"""v4: design the edit on a broad pool, select settings by leave-source-out validation,
judge on held-out halves AND on sources never used in design.

  modal run wsp_v4.py --stage prep
  modal run wsp_v4.py --stage select
  modal run wsp_v4.py --stage final --cfg '{...}'
"""
import json
import modal

VOL = "/vol"
MODEL_DIR = f"{VOL}/base_model"
ROWS = f"{VOL}/v4_rows.json"
FEATS = f"{VOL}/v4_feats.pt"
app = modal.App("wsp-prompt-injection-v4")
volume = modal.Volume.from_name("wsp-prompt-injection")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "transformers==4.45.2", "datasets==3.0.1", "sentencepiece", "protobuf",
                 "scikit-learn", "numpy<2", "pyarrow==17.0.0", "tiktoken")
    .add_local_python_source("wsp_lib", "wsp_path")
)

# leave-source-out folds over the design sources (each fold has attacks and benign text)
FOLDS = [["deepset", "notinject", "oasst_benign"],
         ["safeguard", "ood_roleplay_benign", "email_benign"],
         ["jailbreak_cls", "wild_jailbreak", "wild_regular_benign", "ood_alpaca_benign"],
         ["gandalf", "llmail_inj", "dolly_benign"]]


@app.function(image=image, volumes={VOL: volume}, timeout=3600, cpu=4, memory=16384)
def prep():
    import random
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
    rng = random.Random(0)
    rows, seen, status = [], set(), {}

    def put(ds, text, label, role):
        if not isinstance(text, str) or len(text.strip()) < 15 or text.strip() in seen:
            return False
        seen.add(text.strip())
        rows.append({"ds": ds, "text": text, "label": int(label), "split": role})
        return True

    # ---- everything used so far
    for r in json.load(open(f"{VOL}/final_scores_v2.json"))["base"]:
        if r["ds"] == "ood_jayavibhav":          # synthetic, labels unreliable -> dropped
            continue
        role = {"dev": "design", "dev2": "design", "test": "heldout", "ood": "heldout"}[r["split"]]
        if r["ds"] == "ood_multilingual_inj":
            role = "unseen"                      # never used for design
        put(r["ds"], r["text"], r["label"], role)
    n = 0
    for r in json.load(open(f"{VOL}/eval2_scores_v3.json")):
        n += 1
        put(r["ds"], r["text"], r["label"], "design" if n % 2 == 0 else "heldout")

    # ---- new sources
    def add(name, fn):
        try:
            n0 = len(rows)
            fn()
            status[name] = f"ok: {len(rows) - n0}"
        except Exception as e:  # noqa
            status[name] = f"FAILED {type(e).__name__}: {str(e)[:160]}"
        print(name, status[name])

    def emails():        # benign e-mails: without them "looks like an e-mail" could pass for "attack"
        ds = load_dataset("SetFit/enron_spam", split="train")
        ham = [r["text"] for r in ds if r["label"] == 0 and 100 < len(r["text"]) < 4000]
        rng.shuffle(ham)
        for i, t in enumerate(ham[:3000]):
            put("email_benign", t, 0, "design" if i % 2 == 0 else "heldout")

    def llmail2():       # later phase of the LLMail challenge: never in design
        ds = load_dataset("microsoft/llmail-inject-challenge", split="Phase2", streaming=True)
        got = []
        for r in ds:
            got.append((r.get("subject") or "") + "\n" + (r.get("body") or ""))
            if len(got) >= 20000:
                break
        rng.shuffle(got)
        k = 0
        for t in got:
            k += put("llmail_phase2_inj", t, 1, "unseen")
            if k >= 2000:
                break

    def toxicchat():     # real user traffic with human jailbreak labels: never in design
        got = []
        for sp in ("train", "test"):
            for r in load_dataset("lmsys/toxic-chat", "toxicchat0124", split=sp):
                got.append((r["user_input"], int(r["jailbreaking"])))
        rng.shuffle(got)
        k = 0
        for t, y in got:
            if y == 1:
                put("toxicchat_jailbreak", t, 1, "unseen")
            elif k < 3000:
                k += put("toxicchat_benign", t, 0, "unseen")

    def tensortrust():   # human-written attacks from the Tensor Trust game: never in design
        files = [f for f in list_repo_files("qxcv/tensor-trust", repo_type="dataset")
                 if f.endswith(".jsonl") and "robustness" in f]
        got = []
        for f in files:
            for line in open(hf_hub_download("qxcv/tensor-trust", f, repo_type="dataset"), encoding="utf-8"):
                a = json.loads(line).get("attack")
                if a:
                    got.append(a)
        rng.shuffle(got)
        k = 0
        for t in got:
            k += put("tensortrust_inj", t, 1, "unseen")
            if k >= 2000:
                break

    def rubend():
        for r in load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", split="train"):
            put("rubend_jailbreak", r["Prompt"], 1, "unseen")

    def ultrachat():     # benign first user turns: never in design
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft").shuffle(seed=0).select(range(2000))
        for r in ds:
            put("ultrachat_benign", r["prompt"], 0, "unseen")

    for nme, f in [("email_benign", emails), ("llmail_phase2", llmail2), ("toxicchat", toxicchat),
                   ("tensortrust", tensortrust), ("rubend", rubend), ("ultrachat", ultrachat)]:
        add(nme, f)
    json.dump(rows, open(ROWS, "w"))
    volume.commit()
    tab = {}
    for r in rows:
        k = f"{r['split']:8s} {r['ds']:22s} y={r['label']}"
        tab[k] = tab.get(k, 0) + 1
    samples = {}
    for ds in ("email_benign", "llmail_phase2_inj", "toxicchat_jailbreak", "toxicchat_benign", "tensortrust_inj",
               "ultrachat_benign"):
        g = [r["text"][:150] for r in rows if r["ds"] == ds]
        samples[ds] = rng.sample(g, min(5, len(g)))
    return {"status": status, "table": dict(sorted(tab.items())), "samples": samples}


def _load_feats():
    """Baseline scores + last-FFN-input CLS state for every row (cached on the volume)."""
    import os
    import torch
    import wsp_lib as L
    rows = json.load(open(ROWS))
    if os.path.exists(FEATS):
        f = torch.load(FEATS)
        if len(f["p"]) == len(rows):
            return rows, f["hp"], f["p"]
    tok, model = L.load(MODEL_DIR)
    texts = [r["text"] for r in rows]
    grabbed = {}
    hk = model.deberta.encoder.layer[-1].attention.register_forward_hook(
        lambda m, i, o: grabbed.__setitem__(0, (o[0] if isinstance(o, tuple) else o)[:, 0].float().cpu()))
    hp = torch.zeros(len(texts), model.config.hidden_size)
    p = torch.zeros(len(texts))
    with torch.no_grad():
        for idx, enc in L.batches(tok, texts):
            lg = model(**enc).logits.float()
            ii = torch.tensor(idx)
            hp[ii] = grabbed[0]
            p[ii] = torch.softmax(lg, -1)[:, 1].cpu()
    hk.remove()
    torch.save({"hp": hp, "p": p}, FEATS)
    volume.commit()
    return rows, hp, p


@app.function(image=image, volumes={VOL: volume}, gpu="A10G", cpu=8, memory=32768, timeout=10800)
def select(grid: list):
    """Leave-source-out validation: every design source is scored by an edit that never saw that source."""
    import torch
    import wsp_lib as L
    import wsp_path as PT
    from safetensors.torch import load_file
    torch.set_num_threads(8)
    rows, hp, p0 = _load_feats()
    P0 = PT.get_params(load_file(f"{MODEL_DIR}/model.safetensors"))
    y = torch.tensor([r["label"] for r in rows])
    design = torch.tensor([r["split"] == "design" for r in rows])
    fold_of = {ds: k for k, f in enumerate(FOLDS) for ds in f}
    missing = sorted({r["ds"] for r in rows if r["split"] == "design"} - set(fold_of))
    fold = torch.tensor([fold_of.get(r["ds"], -1) for r in rows])
    drows = [r for r in rows if r["split"] == "design"]
    didx = [i for i, r in enumerate(rows) if r["split"] == "design"]
    out = {"missing_from_folds": missing,
           "base": L.report([dict(r, p=float(p0[i])) for i, r in zip(didx, drows)], "design"), "rows": []}
    for cfg in grid:
        p = p0.clone()
        for k in range(len(FOLDS)):
            fit, ev = design & (fold != k), design & (fold == k)
            P, _ = PT.design_path(hp, y, fit, P0, cfg)
            p[ev] = torch.softmax(PT.path_forward(hp[ev], P), -1)[:, 1]
        rep = L.report([dict(r, p=float(p[i])) for i, r in zip(didx, drows)], "design")
        out["rows"].append({"cfg": cfg, "rep": rep})
        print(cfg, rep["MACRO"], rep["ALL"]["tp"], rep["ALL"]["fp"])
    return out


@app.function(image=image, volumes={VOL: volume}, gpu="A10G", cpu=8, memory=32768, timeout=10800)
def final(cfg: dict, version: str = "v4"):
    import random
    import torch
    import wsp_lib as L
    import wsp_path as PT
    rows, hp, p0 = _load_feats()
    for i, r in enumerate(rows):
        r["p"] = float(p0[i])
    tok, model = L.load(MODEL_DIR)
    sd = model.state_dict()
    P0 = PT.get_params({k: v.cpu() for k, v in sd.items()})
    y = torch.tensor([r["label"] for r in rows])
    fit = torch.tensor([r["split"] == "design" for r in rows])
    P, info = PT.design_path(hp, y, fit, P0, cfg)
    n_changed = 0
    with torch.no_grad():
        for k, name in PT.K.items():
            new = P[k].to(sd[name].device, sd[name].dtype)
            n_changed += int((new != sd[name]).sum())
            sd[name].copy_(new)
    edit_dir = f"{VOL}/edited_model_{version}"
    model.save_pretrained(edit_dir)
    tok.save_pretrained(edit_dir)
    json.dump({"cfg": cfg, "info": info, "weights_changed": n_changed}, open(f"{edit_dir}/wsp_edit.json", "w"), indent=1)
    del model
    tok, model = L.load(edit_dir)
    p_new = L.score(tok, model, [r["text"] for r in rows])
    edit = [dict(r, p=p_new[i]) for i, r in enumerate(rows)]
    json.dump({"base": rows, "edit": edit}, open(f"{VOL}/final_scores_{version}.json", "w"))
    volume.commit()
    out = {"info": info, "weights_changed": n_changed}
    rng = random.Random(1)
    for sp in ("design", "heldout", "unseen"):
        b = [r for r in rows if r["split"] == sp]
        e = [r for r in edit if r["split"] == sp]
        neg = sorted((r["p"] for r in b if r["label"] == 0), reverse=True)
        fp_e = sum(1 for r in e if r["label"] == 0 and r["p"] >= 0.5)
        thr = neg[fp_e] if fp_e < len(neg) else 0.0
        out[sp] = {"base": L.report(rows, sp), "edit": L.report(edit, sp),
                   "base_at_matched_fp": L.report([dict(r, p=1.0 if r["p"] > thr else 0.0) for r in b], sp)}
        flips = [(r["ds"], r["label"], round(r["p"], 2), round(q["p"], 2), r["text"][:140])
                 for r, q in zip(b, e) if (r["p"] >= .5) != (q["p"] >= .5)]
        out[sp]["flips"] = rng.sample(flips, min(40, len(flips)))
    return out


@app.function(image=image, volumes={VOL: volume}, gpu="A10G", timeout=3600)
def bake(shift: float, src: str = "v4", dst: str = "v5"):
    """Spend part of the v4 gain on fewer false positives: move the classifier bias `shift` logits towards SAFE."""
    import torch
    import wsp_lib as L
    rows = json.load(open(f"{VOL}/final_scores_{src}.json"))["base"]
    tok, model = L.load(f"{VOL}/edited_model_{src}")
    with torch.no_grad():
        model.classifier.bias[1] -= shift / 2
        model.classifier.bias[0] += shift / 2
    model.save_pretrained(f"{VOL}/edited_model_{dst}")
    tok.save_pretrained(f"{VOL}/edited_model_{dst}")
    info = json.load(open(f"{VOL}/edited_model_{src}/wsp_edit.json"))
    info["classifier_bias_shift_towards_safe"] = shift
    json.dump(info, open(f"{VOL}/edited_model_{dst}/wsp_edit.json", "w"), indent=1)
    del model
    tok, model = L.load(f"{VOL}/edited_model_{dst}")
    p = L.score(tok, model, [r["text"] for r in rows])
    edit = [dict(r, p=p[i]) for i, r in enumerate(rows)]
    json.dump({"base": rows, "edit": edit}, open(f"{VOL}/final_scores_{dst}.json", "w"))
    volume.commit()
    return {sp: {"base": L.report(rows, sp), "edit": L.report(edit, sp)} for sp in ("design", "heldout", "unseen")}


@app.local_entrypoint()
def main(stage: str = "prep", cfg: str = "", version: str = "v4"):
    import itertools
    import wsp_lib as L
    if stage == "prep":
        out = prep.remote()
        json.dump(out, open("results_v4_prep.json", "w"), indent=1)
        print(json.dumps(out["status"], indent=1))
        for k, v in out["table"].items():
            print(k, v)
    elif stage == "select":
        grid = []
        for lam, fnq, fpq, rounds in itertools.product([0.03, 0.1, 0.3], [0.99, 0.995, 0.999], [0.99, 0.995], [1, 2, 3]):
            grid.append({"lam": lam, "fn_q": fnq, "fp_q": fpq, "fn_delta": 16, "fp_delta": 16,
                         "rounds": rounds, "order": ["fn", "fp"]})
        out = select.remote(grid)
        json.dump(out, open("results_v4_select.json", "w"), indent=1)
        b = out["base"]
        print("missing from folds:", out["missing_from_folds"])
        print(f"BASE                      macroTPR {100*b['MACRO']['tpr']:.1f} macroFPR {100*b['MACRO']['fpr']:.1f}"
              f"  tp {b['ALL']['tp']} fp {b['ALL']['fp']} auroc {b['ALL']['auroc']:.4f}")
        for r in out["rows"]:
            c, rep = r["cfg"], r["rep"]
            print(f"lam {c['lam']:<5} fnq {c['fn_q']:<6} fpq {c['fp_q']:<6} R {c['rounds']} "
                  f"macroTPR {100*rep['MACRO']['tpr']:.1f} macroFPR {100*rep['MACRO']['fpr']:.1f}"
                  f"  tp {rep['ALL']['tp']} fp {rep['ALL']['fp']} auroc {rep['ALL']['auroc']:.4f}")
    elif stage == "bake":
        out = bake.remote(float(cfg))
        json.dump(out, open("results_final_v5.json", "w"), indent=1)
        for sp in ("heldout", "unseen"):
            for m in ("base", "edit"):
                print(f"== {sp} / {m}")
                print(L.fmt(out[sp][m]))
    elif stage == "final":
        out = final.remote(json.loads(cfg), version)
        json.dump(out, open(f"results_final_{version}.json", "w"), indent=1)
        print("weights changed", out["weights_changed"], json.dumps(out["info"]["units"]))
        for sp in ("design", "heldout", "unseen"):
            for m in ("base", "edit", "base_at_matched_fp"):
                print(f"== {sp} / {m}")
                print(L.fmt(out[sp][m]))
