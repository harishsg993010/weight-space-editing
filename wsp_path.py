"""Last-layer path edit: gated FFN units -> dedicated residual 'wire' coordinate -> classifier.

Everything here is closed-form (means, covariances, quantiles). No gradients, no training loop.
"""
from wsp_lib import lda_dir, gated_unit

PFX = "deberta.encoder.layer.11."
K = {"Wi": PFX + "intermediate.dense.weight", "bi": PFX + "intermediate.dense.bias",
     "Wo": PFX + "output.dense.weight", "bo": PFX + "output.dense.bias",
     "g": PFX + "output.LayerNorm.weight", "be": PFX + "output.LayerNorm.bias",
     "Wp": "pooler.dense.weight", "bp": "pooler.dense.bias",
     "W": "classifier.weight", "b": "classifier.bias"}


def get_params(sd):
    return {k: sd[n].float().clone() for k, n in K.items()}


def path_forward(hp, P, eps=1e-7, ret_all=False):
    """Exact last-FFN -> LayerNorm -> pooler -> classifier computation for the CLS token.

    hp = CLS state after the last layer's attention block (the input of the last FFN)."""
    import torch
    gelu = torch.nn.functional.gelu
    a = gelu(hp @ P["Wi"].T + P["bi"])
    x = hp + a @ P["Wo"].T + P["bo"]
    mu = x.mean(-1, keepdim=True)
    sd = ((x - mu) ** 2).mean(-1, keepdim=True).add(eps).sqrt()
    h = (x - mu) / sd * P["g"] + P["be"]
    lg = gelu(h @ P["Wp"].T + P["bp"]) @ P["W"].T + P["b"]
    return (lg, a, x, sd, h) if ret_all else lg


def install_wire(P, hp, amp=0.25, avoid=()):
    """Free one residual coordinate i in the last layer and turn it into a clean signal wire.

    After this, final-FFN units can write a number D onto coordinate i and the classifier's
    logit difference moves by ~D; with nothing written the model behaves as before.
    """
    import torch
    P = {k: v.clone() for k, v in P.items()}
    _, a, x, sd, h = path_forward(hp, P, ret_all=True)
    wd = P["W"][1] - P["W"][0]
    # wire coordinate: the one whose removal from the pooler disturbs the logit least
    sens = (P["Wp"] * wd[:, None]).abs().sum(0)
    score = h.std(0) * sens
    for a_ in avoid:
        score[a_] = float("inf")
    i = int(torch.argmin(score))
    # FFN / pooler units that matter least -> reprogrammable
    ffn_free = torch.argsort(a.abs().mean(0) * P["Wo"].norm(dim=0)).tolist()
    pool = torch.nn.functional.gelu(h @ P["Wp"].T + P["bp"])
    pool_free = torch.argsort((pool * wd).abs().mean(0)).tolist()
    # 1) nobody else writes to coordinate i; a +/- unit pair cancels what the residual carries there
    P["Wo"][i, :] = 0
    f1, f2 = ffn_free[0], ffn_free[1]
    c_h = hp[:, i].mean().item()
    e = torch.zeros(hp.shape[1])
    e[i] = 1.0
    P["Wi"][f1], P["bi"][f1] = e, -c_h
    P["Wi"][f2], P["bi"][f2] = -e, c_h
    P["Wo"][:, f1], P["Wo"][:, f2] = -e, e                 # gelu(z) - gelu(-z) = z exactly
    P["bo"][i] = -c_h + x.mean().item()                    # park the wire at the LayerNorm mean
    # 2) the pooler stops reading coordinate i except through a dedicated +/- unit pair
    P["bp"] += P["Wp"][:, i] * h[:, i].mean()
    P["Wp"][:, i] = 0
    _, _, _, sd2, h2 = path_forward(hp, P, ret_all=True)
    mu0, sbar = h2[:, i].mean().item(), sd2.mean().item()
    m = amp * sbar                                         # wire amplitude per unit of logit
    c = sbar / (P["g"][i].item() * m)                      # classifier gain that undoes it
    k1, k2 = pool_free[0], pool_free[1]
    P["Wp"][k1], P["bp"][k1] = e, -mu0
    P["Wp"][k2], P["bp"][k2] = -e, mu0
    P["W"][:, k1] = torch.tensor([-c / 2, c / 2])
    P["W"][:, k2] = torch.tensor([c / 2, -c / 2])
    info = {"wire": i, "m": m, "ffn_free": ffn_free[2:], "cancel_units": [f1, f2], "pool_units": [k1, k2],
            "gamma_i": P["g"][i].item(), "sbar": sbar}
    return P, info


def add_path_unit(P, info, n, row, bias, signed_gain):
    """Program free FFN unit #n: fires as gelu(row.h' + bias); moves the logit diff by signed_gain * activation."""
    j = info["ffn_free"][n]
    P["Wi"][j], P["bi"][j] = row, bias
    P["Wo"][:, j] = 0
    P["Wo"][info["wire"], j] = signed_gain * info["m"]
    return j


def design_path(hp, y, fit, P0, cfg, avoid=(), keep_free=False):
    """Greedy, gradient-free construction of the edit from the examples in mask `fit`.

    Each step re-reads the current errors on `fit`, and programs one unit:
      fn unit: fires on still-missed injections, threshold keeps it off for benign text
      fp unit: fires on still-flagged benign text, threshold keeps it off for injections
    """
    P, info = install_wire(P0, hp[fit], cfg.get("amp", 0.25), avoid)
    pos, neg = fit & (y == 1), fit & (y == 0)
    n, log = 0, []
    for _ in range(cfg["rounds"]):
        for kind in cfg["order"]:
            lg = path_forward(hp, P)
            pred = lg[:, 1] > lg[:, 0]
            fire, protect, sign = (pos & ~pred, neg, 1) if kind == "fn" else (neg & pred, pos, -1)
            delta = cfg[kind + "_delta"]
            if not delta or fire.sum() < 10:
                continue
            v = lda_dir(hp[fire], hp[protect], cfg["lam"])
            row, bias, g = gated_unit(hp[fire], hp[protect], v, cfg[kind + "_q"], delta)
            j = add_path_unit(P, info, n, row, bias, sign * g)
            log.append({"kind": kind, "ffn_unit": j, "n_fire": int(fire.sum()), "gain": sign * g})
            n += 1
    info["units"] = log
    info["n_units"] = n
    if not keep_free:
        info.pop("ffn_free")
    return P, info
