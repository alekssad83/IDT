"""
IDT GPU TEST SUITE
==================
Содержит два теста:
TEST A — IDT-CIFAR-01: single-run audit (blinded placebo, 500 perms)
TEST B — IDT-NN-02: G-inversion matched-state test (bs=32 vs bs=512)
Запуск:
  pip install torch torchvision numpy scipy
  python idt_gpu_tests.py
Параметры (менять только здесь, до запуска):
"""
# ═══════════════════════════════════════════════════════════════════
# ЗАФИКСИРОВАННЫЕ ПАРАМЕТРЫ — не менять после начала запуска
# ═══════════════════════════════════════════════════════════════════
CFG = {
    "model": "resnet20",
    "dataset": "cifar10",
    "data_dir": "./data",
    "out_dir": "./idt_results",
    "epochs": 160,
    "seeds": 10,
    "bs_main": 128,
    "lr_main": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "lr_milestones": [80, 120],
    "lr_gamma": 0.1,
    "bs_small": 32,
    "bs_large": 512,
    "lr_small": 0.025,
    "lr_large": 0.200,
    "chk_every": 5,
    "T_min": 25,
    "R": 50,
    "N_fim": 200,
    "hess_batch": 256,
    "reg": 1e-5,
    "N_perm": 500,
    "delta_loss": 5,
    "eps_slow": 1e-8,
    "loss_tol": 0.01,
    "h5_tol": 0.05,
    "delta_L_nn02": 0.02,
}

import os
import sys
import json
import time
import argparse
import numpy as np
from scipy.linalg import eigh
from scipy.stats import spearmanr
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as T
except ImportError:
    print("ERROR: pip install torch torchvision")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cpu":
    print("WARNING: CPU mode — это займёт очень долго. Используй GPU.")


# ═══════════════════════════════════════════════════════════════════
# ResNet-20
# ═══════════════════════════════════════════════════════════════════
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        return torch.relu(self.net(x) + self.skip(x))


class ResNet20(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        self.layer1 = self._make(16, 16, 3, 1)
        self.layer2 = self._make(16, 32, 3, 2)
        self.layer3 = self._make(32, 64, 3, 2)
        self.fc = nn.Linear(64, 10)

    def _make(self, ic, oc, n, s):
        return nn.Sequential(
            BasicBlock(ic, oc, s), *[BasicBlock(oc, oc) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.fc(x.mean([2, 3]))


# ═══════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════
def get_loaders(bs, data_dir):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    tr = T.Compose([
        T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize(mean, std),
    ])
    te = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train_ds = torchvision.datasets.CIFAR10(
        data_dir, True, download=True, transform=tr
    )
    test_ds = torchvision.datasets.CIFAR10(
        data_dir, False, download=True, transform=te
    )
    kw = dict(num_workers=2, pin_memory=True)
    train_l = torch.utils.data.DataLoader(train_ds, bs, shuffle=True, **kw)
    test_l = torch.utils.data.DataLoader(test_ds, 256, shuffle=False, **kw)
    idx = np.random.choice(len(train_ds), 2000, replace=False)
    pool_X = torch.stack([train_ds[i][0] for i in idx])
    pool_y = torch.tensor([train_ds[i][1] for i in idx])
    return train_l, test_l, pool_X, pool_y


# ═══════════════════════════════════════════════════════════════════
# HESSIAN-VECTOR PRODUCT
# ═══════════════════════════════════════════════════════════════════
def hvp(model, loss_fn, X, y, v, params):
    model.zero_grad()
    loss = loss_fn(model(X), y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    gf = torch.cat([g.contiguous().view(-1) for g in grads])
    Hv = torch.autograd.grad(gf @ v, params, retain_graph=False)
    model.zero_grad()
    return torch.cat([h.contiguous().view(-1) for h in Hv]).detach()


# ═══════════════════════════════════════════════════════════════════
# LANCZOS
# ═══════════════════════════════════════════════════════════════════
def lanczos(Av, n, R=50, extra=10):
    k = min(R + extra, n - 2)
    V = torch.zeros(n, k + 1, device=device)
    alpha = torch.zeros(k, device=device)
    beta = torch.zeros(k, device=device)
    v = torch.randn(n, device=device)
    v /= v.norm()
    V[:, 0] = v
    k_act = k
    for j in range(k):
        w = Av(V[:, j])
        alpha[j] = float(V[:, j] @ w)
        w = w - alpha[j] * V[:, j]
        if j > 0:
            w = w - beta[j - 1] * V[:, j - 1]
        for i in range(j + 1):
            w = w - (V[:, i] @ w) * V[:, i]
        b = w.norm().item()
        if b < 1e-10 or j == k - 1:
            k_act = j + 1
            break
        beta[j] = b
        V[:, j + 1] = w / b
    T = (
        torch.diag(alpha[:k_act])
        + torch.diag(beta[: k_act - 1], 1)
        + torch.diag(beta[: k_act - 1], -1)
    ).cpu().numpy()
    ev, ec = np.linalg.eigh(T)
    pos = ev > 1e-4
    if not pos.any():
        return None, None
    ev = ev[pos]
    ec = ec[:, pos]
    idx = np.argsort(ev)[::-1]
    ev = ev[idx]
    ec = ec[:, idx]
    r = min(R, len(ev))
    Q = V[:, :k_act].cpu().numpy() @ ec[:, :r]
    return Q, ev[:r]


def compute_H_r(Av, Q_np):
    r = Q_np.shape[1]
    H_r = np.zeros((r, r))
    for j in range(r):
        q = torch.tensor(Q_np[:, j], dtype=torch.float32, device=device)
        H_r[:, j] = Q_np.T @ Av(q).cpu().numpy()
    return (H_r + H_r.T) / 2


def compute_G_r(model, loss_fn, pool_X, pool_y, Q_np, N_fim, reg):
    r = Q_np.shape[1]
    Q_t = torch.tensor(Q_np, dtype=torch.float32, device=device)
    params = [p for p in model.parameters() if p.requires_grad]
    idx = np.random.choice(len(pool_y), min(N_fim, len(pool_y)), replace=False)
    grads = []
    for i in idx:
        model.zero_grad()
        xi = pool_X[i : i + 1].to(device)
        yi = pool_y[i : i + 1].to(device)
        loss_fn(model(xi), yi).backward()
        g = torch.cat([p.grad.view(-1) for p in params if p.grad is not None])
        grads.append((Q_t.T @ g).detach().cpu().numpy())
        model.zero_grad()
    G = np.array(grads)
    return G.T @ G / len(grads) + np.eye(r) * reg


# ═══════════════════════════════════════════════════════════════════
# M(ω) — FIXED FORMULA
# ═══════════════════════════════════════════════════════════════════
def M_omega(lam):
    lam = np.sort(np.real(lam))
    lam = lam[lam > 1e-10]
    if len(lam) < 2:
        return np.nan
    return float(np.mean(lam**2) / (lam[0] * np.mean(lam)))


def compute_M(H_r, G_r):
    try:
        return M_omega(np.sort(np.real(eigh(H_r, G_r, eigvals_only=True))))
    except Exception:
        return np.nan


def lam_min_positive(H_r):
    ev = np.linalg.eigvalsh(H_r)
    ev = ev[ev > 1e-10]
    return float(ev[0]) if len(ev) > 0 else 1e-10


# ═══════════════════════════════════════════════════════════════════
# SLOWDOWN METRIC
# ═══════════════════════════════════════════════════════════════════
def slowdown(loss_hist, ep, delta, eps):
    if ep + delta >= len(loss_hist):
        return np.nan
    return 1.0 / (abs(loss_hist[ep + delta] - loss_hist[ep]) + eps)


# ═══════════════════════════════════════════════════════════════════
# TRAIN ONE SEED
# ═══════════════════════════════════════════════════════════════════
def train_seed(seed, bs, lr, epochs, out_dir, label, cfg, pool_X, pool_y):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, test_loader, _, _ = get_loaders(bs, cfg["data_dir"])
    model = ResNet20().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
        nesterov=True,
    )
    sched = optim.lr_scheduler.MultiStepLR(
        opt, cfg["lr_milestones"], cfg["lr_gamma"]
    )
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    loss_hist, acc_hist, checkpoints = [], [], []
    iters_total = 0
    print(
        f"\n  [{label}] seed={seed} bs={bs} lr={lr:.4f} n_params={n_params:,}"
    )
    for ep in range(epochs):
        model.train()
        ep_loss = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(Xb), yb).backward()
            opt.step()
            ep_loss.append(loss_fn(model(Xb), yb).item())
            iters_total += 1
        loss_hist.append(float(np.mean(ep_loss)))
        sched.step()
        if ep % 10 == 0:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for Xb, yb in test_loader:
                    pred = model(Xb.to(device)).argmax(1)
                    correct += (pred == yb.to(device)).sum().item()
                    total += len(yb)
            acc = correct / total
            acc_hist.append((ep, acc))
            print(
                f"  ep={ep:>3} loss={loss_hist[-1]:.4f} "
                f"acc={acc:.3f} iters={iters_total}"
            )
        if ep >= cfg["chk_every"] and ep % cfg["chk_every"] == 0:
            model.eval()
            torch.manual_seed(seed * 10000 + ep)
            np.random.seed(seed * 10000 + ep)
            idx_h = np.random.choice(
                len(pool_y), cfg["hess_batch"], replace=False
            )
            Xh = pool_X[idx_h].to(device)
            yh = pool_y[idx_h].to(device)

            def Av(v):
                return hvp(model, loss_fn, Xh, yh, v, params)

            Q, lam_H = lanczos(Av, n_params, R=cfg["R"])
            if Q is None or len(lam_H) < 5:
                continue
            H_r = compute_H_r(Av, Q)
            np.random.seed(seed * 10000 + ep + 1)
            G_r = compute_G_r(
                model, loss_fn, pool_X, pool_y, Q, cfg["N_fim"], cfg["reg"]
            )
            checkpoints.append({
                "ep": ep,
                "iters": iters_total,
                "loss": loss_hist[-1],
                "H_r": H_r,
                "G_r": G_r,
                "lam_H": lam_H,
                "top5_H": lam_H[-5:],
            })
            print(
                f"  checkpoint ep={ep} r={len(lam_H)} "
                f"lam_min_H={lam_H[-1]:.5f} M={compute_M(H_r, G_r):.2f}"
            )
    save = Path(out_dir) / f"seed{seed}_{label}.npz"
    np.savez_compressed(
        str(save),
        loss_hist=np.array(loss_hist),
        acc_hist=np.array(acc_hist),
        ep_arr=np.array([c["ep"] for c in checkpoints]),
        iter_arr=np.array([c["iters"] for c in checkpoints]),
        loss_arr=np.array([c["loss"] for c in checkpoints]),
        top5_arr=np.array([c["top5_H"] for c in checkpoints]),
        H_r_arr=np.array([c["H_r"] for c in checkpoints], dtype=object),
        G_r_arr=np.array([c["G_r"] for c in checkpoints], dtype=object),
        lam_H_arr=np.array([c["lam_H"] for c in checkpoints], dtype=object),
    )
    print(f"  Saved {len(checkpoints)} checkpoints → {save}")
    return checkpoints, loss_hist


# ═══════════════════════════════════════════════════════════════════
# TEST A — IDT-CIFAR-01
# ═══════════════════════════════════════════════════════════════════
def run_audit_A(checkpoints, loss_hist, cfg, seed_id):
    T = len(checkpoints)
    if T < 5:
        return None
    tau = np.array([
        slowdown(loss_hist, c["ep"], cfg["delta_loss"], cfg["eps_slow"])
        for c in checkpoints
    ])
    valid_tau = ~np.isnan(tau)
    P_H = np.array([1.0 / lam_min_positive(c["H_r"]) for c in checkpoints])
    P_M = np.array([compute_M(c["H_r"], c["G_r"]) for c in checkpoints])
    valid = valid_tau & ~np.isnan(P_M)
    if valid.sum() < 5:
        return None
    rho_H, _ = spearmanr(tau[valid], P_H[valid])
    rho_M, _ = spearmanr(tau[valid], P_M[valid])
    rng = np.random.default_rng(seed_id * 9999)
    perm_rhos = []
    for _ in range(cfg["N_perm"]):
        pi = rng.permutation(T)
        Pp = np.array([
            compute_M(checkpoints[t]["H_r"], checkpoints[pi[t]]["G_r"])
            for t in range(T)
        ])
        v2 = valid_tau & ~np.isnan(Pp)
        if v2.sum() < 5:
            continue
        r, _ = spearmanr(tau[v2], Pp[v2])
        perm_rhos.append(r)
    perm_rhos = np.array(perm_rhos)
    pctile = float((perm_rhos < rho_M).mean() * 100)
    return {
        "T": T,
        "n_valid": int(valid.sum()),
        "rho_H": rho_H,
        "rho_M": rho_M,
        "delta_rho": rho_M - rho_H,
        "perm_mean": float(perm_rhos.mean()),
        "perm_p95": float(np.percentile(perm_rhos, 95)),
        "perm_p99": float(np.percentile(perm_rhos, 99)),
        "pctile_M": pctile,
        "beats_H": bool(rho_M > rho_H),
        "above_p95": bool(rho_M > np.percentile(perm_rhos, 95)),
        "null_ok": bool(abs(perm_rhos.mean()) < 0.15),
    }


# ═══════════════════════════════════════════════════════════════════
# TEST B — IDT-NN-02
# ═══════════════════════════════════════════════════════════════════
def tau_to_delta(loss_hist, iter_hist, from_ep_idx, delta_L):
    target = loss_hist[from_ep_idx] - delta_L
    for k in range(from_ep_idx + 1, len(loss_hist)):
        if loss_hist[k] <= target:
            return float(iter_hist[k] - iter_hist[from_ep_idx])
    return np.nan


def run_ginv_B(ck_small, loss_small, ck_large, loss_large, cfg):
    matched = []
    ep_small = [c["ep"] for c in ck_small]
    it_small = [c["iters"] for c in ck_small]
    ep_large = [c["ep"] for c in ck_large]
    it_large = [c["iters"] for c in ck_large]
    for i, ci in enumerate(ck_small):
        for j, cj in enumerate(ck_large):
            if abs(ci["loss"] - cj["loss"]) > cfg["loss_tol"]:
                continue
            h5i, h5j = ci["top5_H"], cj["top5_H"]
            rel = np.abs(h5i - h5j) / (np.abs(h5j) + 1e-8)
            if rel.max() > cfg["h5_tol"]:
                continue
            tau_s = tau_to_delta(loss_small, it_small, i, cfg["delta_L_nn02"])
            tau_l = tau_to_delta(loss_large, it_large, j, cfg["delta_L_nn02"])
            if np.isnan(tau_s) or np.isnan(tau_l):
                continue
            M_s = compute_M(ci["H_r"], ci["G_r"])
            M_l = compute_M(cj["H_r"], cj["G_r"])
            MH_s = 1.0 / lam_min_positive(ci["H_r"])
            MH_l = 1.0 / lam_min_positive(cj["H_r"])
            if np.isnan(M_s) or np.isnan(M_l):
                continue
            matched.append({
                "loss_s": ci["loss"], "loss_l": cj["loss"],
                "tau_s": tau_s, "tau_l": tau_l,
                "M_s": M_s, "M_l": M_l,
                "MH_s": MH_s, "MH_l": MH_l,
                "h5_diff": float(rel.max()),
            })
    if len(matched) < 3:
        return {"n_matched": len(matched), "status": "too_few_pairs"}
    tau_ratios = [m["tau_s"] / m["tau_l"] for m in matched]
    M_ratios = [m["M_s"] / m["M_l"] for m in matched]
    MH_ratios = [m["MH_s"] / m["MH_l"] for m in matched]
    tau_ratio = float(np.median(tau_ratios))
    M_ratio = float(np.median(M_ratios))
    MH_ratio = float(np.median(MH_ratios))
    sign_acc = sum(
        1 for m in matched
        if (m["M_s"] > m["M_l"]) == (m["tau_s"] > m["tau_l"])
    ) / len(matched)
    rho_M = np.nan
    if len(matched) >= 5:
        rho_M, _ = spearmanr(M_ratios, tau_ratios)
    idt_pass = (tau_ratio > 1.3) and (sign_acc > 0.6)
    return {
        "n_matched": len(matched),
        "tau_ratio": tau_ratio,
        "M_ratio": M_ratio,
        "MH_ratio": MH_ratio,
        "sign_acc": sign_acc,
        "rho_M": float(rho_M) if not np.isnan(rho_M) else None,
        "idt_pass": idt_pass,
        "H_matched": bool(abs(MH_ratio - 1.0) < 0.2),
    }


# ═══════════════════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════════════════
def verdict_A(reports):
    n = len(reports)
    beats_H = sum(r["beats_H"] for r in reports)
    above_p95 = sum(r["above_p95"] for r in reports)
    mean_pct = float(np.mean([r["pctile_M"] for r in reports]))
    mean_delt = float(np.mean([r["delta_rho"] for r in reports]))
    c1 = beats_H >= max(5, int(n * 0.6))
    c2 = above_p95 >= max(5, int(n * 0.6))
    c3 = mean_pct >= 90.0
    c4 = mean_delt >= 0.05
    n_pass = sum([c1, c2, c3, c4])
    v = {4: "FULL PASS", 3: "PARTIAL PASS", 2: "AMBIGUOUS", 1: "FAIL", 0: "FAIL"}
    return {
        "C1_beats_H": (beats_H, n, c1),
        "C2_above_p95": (above_p95, n, c2),
        "C3_pctile": (mean_pct, c3),
        "C4_delta_rho": (mean_delt, c4),
        "n_pass": n_pass,
        "VERDICT": v[n_pass],
    }


def verdict_B(results):
    n = len(results)
    n_pass = sum(r.get("idt_pass", False) for r in results)
    median_tau = float(np.nanmedian([r.get("tau_ratio", np.nan) for r in results]))
    mean_sign = float(np.nanmean([r.get("sign_acc", np.nan) for r in results]))
    mean_rho = float(np.nanmean([
        r["rho_M"] for r in results if r.get("rho_M") is not None
    ])) if any(r.get("rho_M") is not None for r in results) else np.nan
    h_matched = sum(r.get("H_matched", False) for r in results)
    p1 = n_pass >= max(5, int(n * 0.6))
    p2 = mean_sign >= 0.6
    p3 = mean_rho >= 0.3 if not np.isnan(mean_rho) else False
    n_p = sum([p1, p2, p3])
    vdict = {3: "FULL PASS", 2: "PARTIAL PASS", 1: "AMBIGUOUS", 0: "FAIL"}
    return {
        "n_idt_pass": (n_pass, n),
        "median_tau_ratio": median_tau,
        "mean_sign_acc": mean_sign,
        "mean_rho_M": mean_rho,
        "H_matched_seeds": (h_matched, n),
        "VERDICT": vdict[n_p],
    }


def print_A(reports, vdict):
    print("\n" + "=" * 65)
    print("TEST A — IDT-CIFAR-01 РЕЗУЛЬТАТЫ")
    print("=" * 65)
    print(
        f"{'Seed':>5} {'T':>4} {'rho_H':>7} {'rho_M':>7} "
        f"{'perm_mean':>10} {'pctile':>7} {'beats_H':>8} {'>p95':>6}"
    )
    print("-" * 65)
    for r in reports:
        print(
            f"{r.get('seed', 0):>5} {r['T']:>4} {r['rho_H']:>+7.3f} "
            f"{r['rho_M']:>+7.3f} {r['perm_mean']:>+10.3f} "
            f"{r['pctile_M']:>7.1f}% "
            f"{'✓' if r['beats_H'] else '✗':>8} "
            f"{'✓' if r['above_p95'] else '✗':>6}"
        )
    print()
    c1, n, p1 = vdict["C1_beats_H"]
    c2, n, p2 = vdict["C2_above_p95"]
    print(f"C1 M>H:     {c1}/{n}  {'PASS' if p1 else 'FAIL'}")
    print(f"C2 >p95 perm: {c2}/{n}  {'PASS' if p2 else 'FAIL'}")
    print(f"C3 pctile:  {vdict['C3_pctile'][0]:.1f}%  {'PASS' if vdict['C3_pctile'][1] else 'FAIL'}")
    print(f"C4 Δρ:      {vdict['C4_delta_rho'][0]:+.3f}  {'PASS' if vdict['C4_delta_rho'][1] else 'FAIL'}")
    print(f"\nВЕРДИКТ A: {vdict['VERDICT']}")


def print_B(per_seed, vdict):
    print("\n" + "=" * 65)
    print("TEST B — IDT-NN-02 G-INVERSION РЕЗУЛЬТАТЫ")
    print("=" * 65)
    print(
        f"{'Seed':>5} {'n_match':>8} {'tau_ratio':>10} {'M_ratio':>9} "
        f"{'MH_ratio':>9} {'sign%':>7} {'rho_M':>7} {'PASS':>6}"
    )
    print("-" * 70)
    for r in per_seed:
        if "n_matched" not in r:
            continue
        print(
            f"{r.get('seed', 0):>5} {r['n_matched']:>8} "
            f"{r.get('tau_ratio', float('nan')):>10.2f}x "
            f"{r.get('M_ratio', float('nan')):>9.1f}x "
            f"{r.get('MH_ratio', float('nan')):>9.3f} "
            f"{r.get('sign_acc', float('nan')):>7.0%} "
            f"{r['rho_M'] if r.get('rho_M') is not None else float('nan'):>+7.3f} "
            f"{'✓' if r.get('idt_pass') else '✗':>6}"
        )
    print()
    np_, n = vdict["n_idt_pass"]
    print(f"G-inversion PASS: {np_}/{n}")
    print(f"Median tau ratio: {vdict['median_tau_ratio']:.2f}x  (target: 1.5–5×)")
    print(f"Mean sign acc:    {vdict['mean_sign_acc']:.0%}  (target: >60%)")
    print(f"Mean rho(ΔM,Δτ):  {vdict['mean_rho_M']:+.3f}  (target: ≥0.3)")
    hm, n = vdict["H_matched_seeds"]
    print(f"H matched seeds:  {hm}/{n}")
    print(f"\nВЕРДИКТ B: {vdict['VERDICT']}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", choices=["A", "B", "AB"], default="AB",
        help="A=audit only, B=ginv only, AB=both",
    )
    parser.add_argument("--seeds", type=int, default=CFG["seeds"])
    parser.add_argument("--out", type=str, default=CFG["out_dir"])
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Быстрый прогон: 20 эпох, 1 seed (только проверка кода)",
    )
    args = parser.parse_args()
    CFG["seeds"] = args.seeds
    CFG["out_dir"] = args.out
    if args.quick:
        CFG["epochs"] = 20
        CFG["seeds"] = 1
        CFG["chk_every"] = 5
        CFG["N_perm"] = 50
        print("QUICK MODE: 20 epochs, 1 seed, 50 perms")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "config.json", "w") as f:
        json.dump(CFG, f, indent=2)
    print("Config saved. Starting tests...")
    print(f"Tests: {args.test}  |  Seeds: {CFG['seeds']}  |  Epochs: {CFG['epochs']}")
    t_start = time.time()
    reports_A = []
    per_seed_B = []
    for seed in range(CFG["seeds"]):
        print(f"\n{'='*50}")
        print(f"SEED {seed + 1}/{CFG['seeds']}")
        print(f"{'='*50}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        _, _, pool_X, pool_y = get_loaders(CFG["bs_main"], CFG["data_dir"])
        if args.test in ("A", "AB"):
            ck_main, loss_main = train_seed(
                seed, CFG["bs_main"], CFG["lr_main"],
                CFG["epochs"], args.out, "main", CFG, pool_X, pool_y,
            )
            rep = run_audit_A(ck_main, loss_main, CFG, seed)
            if rep:
                rep["seed"] = seed
                reports_A.append(rep)
                print(
                    f"\n  A-audit: rho_H={rep['rho_H']:+.3f} rho_M={rep['rho_M']:+.3f} "
                    f"pctile={rep['pctile_M']:.0f}% "
                    f"{'✓M>H' if rep['beats_H'] else '✗'}"
                )
        if args.test in ("B", "AB"):
            ck_s, loss_s = train_seed(
                seed, CFG["bs_small"], CFG["lr_small"],
                CFG["epochs"], args.out, "small", CFG, pool_X, pool_y,
            )
            ck_l, loss_l = train_seed(
                seed, CFG["bs_large"], CFG["lr_large"],
                CFG["epochs"], args.out, "large", CFG, pool_X, pool_y,
            )
            ginv = run_ginv_B(ck_s, loss_s, ck_l, loss_l, CFG)
            ginv["seed"] = seed
            per_seed_B.append(ginv)
            print(
                f"\n  B-ginv: n_match={ginv['n_matched']} "
                f"tau_ratio={ginv.get('tau_ratio', float('nan')):.2f}x "
                f"MH_ratio={ginv.get('MH_ratio', float('nan')):.3f} "
                f"{'✓PASS' if ginv.get('idt_pass') else '✗FAIL'}"
            )
    elapsed = (time.time() - t_start) / 60
    print(f"\n  Elapsed: {elapsed:.0f} min")
    print(f"\n\n{'='*65}")
    print("ИТОГОВЫЙ ОТЧЁТ")
    print(f"{'='*65}")
    final = {"config": CFG, "seeds_run": CFG["seeds"]}
    if reports_A:
        vA = verdict_A(reports_A)
        print_A(reports_A, vA)
        final["test_A"] = {"per_seed": reports_A, "verdict": vA}
    if per_seed_B:
        vB = verdict_B(per_seed_B)
        print_B(per_seed_B, vB)
        final["test_B"] = {"per_seed": per_seed_B, "verdict": vB}
    print(f"\n{'='*65}")
    print("ОБЩИЙ ВЕРДИКТ")
    print(f"{'='*65}")
    if reports_A and per_seed_B:
        va = vA["VERDICT"]
        vb = vB["VERDICT"]
        if "PASS" in va and "PASS" in vb:
            overall = "FULL PASS — IDT G-активность подтверждена"
        elif "PASS" in va or "PASS" in vb:
            overall = "PARTIAL PASS — один из двух тестов прошёл"
        elif "AMBIGUOUS" in va or "AMBIGUOUS" in vb:
            overall = "AMBIGUOUS — сигнал есть, не решающий"
        else:
            overall = "FAIL — G-активность не подтверждена на этой настройке"
        print(f"Test A: {va}")
        print(f"Test B: {vb}")
        print(f"ИТОГ:   {overall}")
    report_path = Path(args.out) / "final_report.json"
    with open(report_path, "w") as f:
        json.dump(final, f, indent=2, default=float)
    print(f"\nОтчёт сохранён: {report_path}")
    print(f"Общее время: {(time.time() - t_start) / 3600:.1f} ч")


if __name__ == "__main__":
    main()
