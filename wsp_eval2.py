"""Second out-of-distribution eval: human-written sources only, none used when designing the edit.

  modal run wsp_eval2.py
"""
import json
import modal

VOL = "/vol"
app = modal.App("wsp-prompt-injection-eval2")
volume = modal.Volume.from_name("wsp-prompt-injection")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "transformers==4.45.2", "datasets==3.0.1", "sentencepiece", "protobuf",
                 "scikit-learn", "numpy<2", "pyarrow==17.0.0", "tiktoken")
    .add_local_python_source("wsp_lib")
)


def _load(seen):
    import random
    from datasets import load_dataset
    rng = random.Random(0)
    rows, status = [], {}

    def add(name, fn):
        try:
            got = [(t, y) for t, y in fn() if isinstance(t, str) and len(t.strip()) >= 15]
            uniq = []
            for t, y in got:
                k = t.strip()
                if k not in seen:
                    seen.add(k)
                    uniq.append((t, y))
            rng.shuffle(uniq)
            uniq = uniq[:2000]
            rows.extend({"ds": name, "text": t, "label": y, "split": "ood2"} for t, y in uniq)
            status[name] = f"ok: {len(uniq)} rows (of {len(got)} loaded)"
        except Exception as e:  # noqa
            status[name] = f"FAILED {type(e).__name__}: {str(e)[:160]}"
        print(name, status[name])

    # ---- injections / jailbreaks written by real people
    def hackaprompt():  # competition submissions that actually succeeded
        ds = load_dataset("hackaprompt/hackaprompt-dataset", split="train")
        ds = ds.filter(lambda r: bool(r["correct"]) and r["user_input"] is not None)
        ds = ds.shuffle(seed=0).select(range(min(20000, len(ds))))
        return [(r["user_input"], 1) for r in ds]

    def wild_jb():      # jailbreak prompts collected from Reddit / Discord / prompt sites
        ds = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25", split="train")
        return [(r["prompt"], 1) for r in ds]

    def llmail():       # attack e-mails from Microsoft's LLMail-Inject challenge
        ds = load_dataset("microsoft/llmail-inject-challenge", split="Phase1", streaming=True)
        out = []
        for r in ds:
            out.append(((r.get("subject") or "") + "\n" + (r.get("body") or ""), 1))
            if len(out) >= 20000:
                break
        return out

    # ---- benign text written by real people
    def wild_regular():  # ordinary (non-jailbreak) prompts from the same communities: hard negatives
        ds = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", "regular_2023_12_25", split="train")
        return [(r["prompt"], 0) for r in ds]

    def oasst():        # first user messages of OpenAssistant conversations
        ds = load_dataset("OpenAssistant/oasst1", split="train")
        return [(r["text"], 0) for r in ds if r["parent_id"] is None and r["lang"] == "en"]

    def dolly():        # instructions written by Databricks employees
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        return [(r["instruction"], 0) for r in ds]

    for n, f in [("hackaprompt_inj", hackaprompt), ("wild_jailbreak", wild_jb), ("llmail_inj", llmail),
                 ("wild_regular_benign", wild_regular), ("oasst_benign", oasst), ("dolly_benign", dolly)]:
        add(n, f)
    return rows, status


@app.function(image=image, volumes={VOL: volume}, gpu="A10G", timeout=7200)
def run(version: str = "v2"):
    import random
    import torch
    import wsp_lib as L
    prev = json.load(open(f"{VOL}/final_scores_v2.json"))["base"]   # everything touched so far
    rows, status = _load({r["text"].strip() for r in prev})
    texts = [r["text"] for r in rows]
    scores = {}
    for name, path in (("base", f"{VOL}/base_model"), ("edit", f"{VOL}/edited_model_{version}")):
        tok, model = L.load(path)
        scores[name] = L.score(tok, model, texts)
        del model
        torch.cuda.empty_cache()
    for i, r in enumerate(rows):
        r["p_base"], r["p_edit"] = scores["base"][i], scores["edit"][i]
    json.dump(rows, open(f"{VOL}/eval2_scores_{version}.json", "w"))
    volume.commit()
    base = [dict(r, p=r["p_base"]) for r in rows]
    edit = [dict(r, p=r["p_edit"]) for r in rows]
    out = {"status": status, "base": L.report(base, "ood2"), "edit": L.report(edit, "ood2")}
    # baseline re-thresholded so it makes the same number of false positives as the edited model
    neg = sorted((r["p"] for r in base if r["label"] == 0), reverse=True)
    fp_e = sum(1 for r in edit if r["label"] == 0 and r["p"] >= 0.5)
    thr = neg[fp_e] if fp_e < len(neg) else 0.0
    out["base_at_matched_fp"] = L.report([dict(r, p=1.0 if r["p"] > thr else 0.0) for r in base], "ood2")
    out["matched_thr"] = thr
    # samples, to check the labels by eye
    rng = random.Random(1)
    out["samples"] = {}
    for ds in sorted({r["ds"] for r in rows}):
        g = [r for r in rows if r["ds"] == ds]
        out["samples"][ds] = [(round(r["p_base"], 2), round(r["p_edit"], 2), r["text"][:160]) for r in rng.sample(g, 8)]
        flip = [r for r in g if (r["p_base"] >= .5) != (r["p_edit"] >= .5)]
        out["samples"][ds + "__changed"] = [(round(r["p_base"], 2), round(r["p_edit"], 2), r["text"][:160])
                                            for r in rng.sample(flip, min(8, len(flip)))]
    return out


@app.local_entrypoint()
def main(version: str = "v2"):
    import wsp_lib as L
    out = run.remote(version)
    json.dump(out, open(f"results_eval2_{version}.json", "w"), indent=1)
    print(json.dumps(out["status"], indent=1))
    for k in ("base", "edit", "base_at_matched_fp"):
        print("==", k)
        print(L.fmt(out[k]))
