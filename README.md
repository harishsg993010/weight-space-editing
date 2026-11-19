# Weight-space editing a prompt-injection classifier

Gradient-free weight edits to `protectai/deberta-v3-base-prompt-injection-v2`, following the idea in
[Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).

**Write-up:** https://harishsg993010.github.io/weight-space-editing/

| File | What it does |
|---|---|
| `wsp_app.py` | Data prep, baseline, per-layer diagnosis, edits v1 to v3 |
| `wsp_path.py` | The edit itself: wire coordinate, gated units, exact readout |
| `wsp_lib.py` | Scoring, metrics, closed-form directions |
| `wsp_eval2.py` | Human-written out-of-distribution eval |
| `wsp_v4.py` | Broad design pool, leave-one-source-out selection, v4/v5 |
| `wsp_probe.py` | Search for a signal that transfers across sources |
| `wsp_v6.py` | Token-max attention-head circuit |
| `wsp_backprop.py` | Comparison with backprop on the same data (same weights, last layer, full fine-tuning) |
| `wsp_v7.py` | Word-average attention head, weights from least squares |
| `wsp_v8.py` | Version 7 plus a per-word score from layer 10 inner units, margin fit |
| `wsp_upload.py` | Uploads the edited model to Hugging Face |
| `results/` | Numeric results for every stage (no dataset text) |

Everything runs on [Modal](https://modal.com); see the write-up for the commands. The edited model (version 5) is on Hugging Face: https://huggingface.co/rootxhacker/deberta-v3-base-prompt-injection-v2-weight-edited
