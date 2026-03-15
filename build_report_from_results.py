#!/usr/bin/env python3
"""
Строит отчёт VALIDATION_REPORT.md из final_report.json после прогона idt_gpu_tests.py.
Опционально: bootstrap 95% ДИ для ключевых метрик (--ci N).
Использование:
  python build_report_from_results.py --input results/final_report.json --output results/VALIDATION_REPORT.md
  python build_report_from_results.py --input results/final_report.json --output results/VALIDATION_REPORT.md --ci 2000
"""
import argparse
import json
import numpy as np
from pathlib import Path


def load_report(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bootstrap_ci(values, n_bootstrap=2000, ci=0.95):
    """Bootstrap 95% CI for mean of values (resample with replacement)."""
    if not values or len(values) < 2:
        return (np.nan, np.nan)
    n = len(values)
    arr = np.array(values, dtype=float)
    np.random.seed(42)
    means = [np.mean(np.random.choice(arr, size=n, replace=True)) for _ in range(n_bootstrap)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return (lo, hi)


def build_report(data: dict, out_path: Path, ci_bootstrap: int = 0) -> str:
    lines = []
    lines.append("# Отчёт валидации IDT GPU Tests")
    lines.append("")
    lines.append("Источник: `final_report.json` (прогон idt_gpu_tests.py).")
    lines.append("")

    config = data.get("config", {})
    seeds_run = data.get("seeds_run", config.get("seeds", "?"))
    lines.append(f"**Конфигурация:** seeds={seeds_run}, epochs={config.get('epochs', '?')}, "
                f"dataset={config.get('dataset', '?')}, model={config.get('model', '?')}.")
    lines.append("")

    # Test A
    if "test_A" in data:
        tA = data["test_A"]
        per_seed = tA.get("per_seed", [])
        vA = tA.get("verdict", {})
        lines.append("## Test A — IDT-CIFAR-01 (blinded placebo audit)")
        lines.append("")
        lines.append("| Seed | T | ρ_H | ρ_M | perm_mean | pctile_M | beats_H | >p95 |")
        lines.append("|------|---|-----|-----|-----------|----------|--------|-----|")
        for r in per_seed:
            lines.append(
                f"| {r.get('seed', '')} | {r.get('T', '')} | "
                f"{r.get('rho_H', 0):+.3f} | {r.get('rho_M', 0):+.3f} | "
                f"{r.get('perm_mean', 0):+.3f} | {r.get('pctile_M', 0):.1f}% | "
                f"{'✓' if r.get('beats_H') else '✗'} | {'✓' if r.get('above_p95') else '✗'} |"
            )
        lines.append("")
        c1, n, p1 = vA.get("C1_beats_H", (0, 0, False))
        c2, n, p2 = vA.get("C2_above_p95", (0, 0, False))
        lines.append("**Критерии:**")
        lines.append(f"- C1 M>H: {c1}/{n} — " + ("PASS" if p1 else "FAIL"))
        lines.append(f"- C2 >p95 perm: {c2}/{n} — " + ("PASS" if p2 else "FAIL"))
        lines.append(f"- C3 pctile ≥90%: {vA.get('C3_pctile', (0, False))[0]:.1f}% — " +
                    ("PASS" if vA.get("C3_pctile", (0, False))[1] else "FAIL"))
        lines.append(f"- C4 Δρ ≥0.05: {vA.get('C4_delta_rho', (0, False))[0]:+.3f} — " +
                    ("PASS" if vA.get("C4_delta_rho", (0, False))[1] else "FAIL"))
        lines.append("")
        lines.append(f"**Вердикт A:** {vA.get('VERDICT', '—')}")
        if ci_bootstrap > 0 and per_seed:
            pctiles = [r["pctile_M"] for r in per_seed]
            delta_rhos = [r["delta_rho"] for r in per_seed]
            lo_p, hi_p = bootstrap_ci(pctiles, n_bootstrap=ci_bootstrap)
            lo_d, hi_d = bootstrap_ci(delta_rhos, n_bootstrap=ci_bootstrap)
            lines.append("")
            lines.append(f"**Bootstrap 95% ДИ** (n={ci_bootstrap}):")
            lines.append(f"- pctile_M: [{lo_p:.1f}%, {hi_p:.1f}%]")
            lines.append(f"- Δρ: [{lo_d:+.3f}, {hi_d:+.3f}]")
        lines.append("")

    # Test B
    if "test_B" in data:
        tB = data["test_B"]
        per_seed = tB.get("per_seed", [])
        vB = tB.get("verdict", {})
        lines.append("## Test B — IDT-NN-02 (G-inversion)")
        lines.append("")
        lines.append("| Seed | n_matched | tau_ratio | M_ratio | MH_ratio | sign_acc | ρ_M | PASS |")
        lines.append("|------|-----------+-----------+---------|----------|----------|-----|------|")
        for r in per_seed:
            if "n_matched" not in r:
                continue
            tau = r.get("tau_ratio", np.nan)
            M = r.get("M_ratio", np.nan)
            MH = r.get("MH_ratio", np.nan)
            sa = r.get("sign_acc", np.nan)
            rho = r.get("rho_M") if r.get("rho_M") is not None else np.nan
            rho_str = f"{rho:+.3f}" if not (isinstance(rho, float) and np.isnan(rho)) else "—"
            lines.append(
                f"| {r.get('seed', '')} | {r.get('n_matched', '')} | "
                f"{tau:.2f}x | {M:.1f}x | {MH:.3f} | {sa:.0%} | "
                f"{rho_str} | "
                f"{'✓' if r.get('idt_pass') else '✗'} |"
            )
        lines.append("")
        np_pass, n = vB.get("n_idt_pass", (0, 0))
        lines.append("**Критерии:**")
        lines.append(f"- G-inversion PASS: {np_pass}/{n}")
        lines.append(f"- Median tau_ratio: {vB.get('median_tau_ratio', np.nan):.2f}x (цель 1.5–5×)")
        lines.append(f"- Mean sign_acc: {vB.get('mean_sign_acc', np.nan):.0%} (цель >60%)")
        rho_m = vB.get("mean_rho_M", np.nan)
        rho_m_str = f"{rho_m:+.3f}" if not (isinstance(rho_m, float) and np.isnan(rho_m)) else "—"
        lines.append(f"- Mean ρ(M_ratio, tau_ratio): {rho_m_str} (цель ≥0.3)")
        lines.append("")
        lines.append(f"**Вердикт B:** {vB.get('VERDICT', '—')}")
        if ci_bootstrap > 0 and per_seed:
            tau_ratios = [r.get("tau_ratio") for r in per_seed if "tau_ratio" in r and not np.isnan(r.get("tau_ratio", np.nan))]
            sign_accs = [r.get("sign_acc") for r in per_seed if "sign_acc" in r and not np.isnan(r.get("sign_acc", np.nan))]
            if tau_ratios:
                lo_t, hi_t = bootstrap_ci(tau_ratios, n_bootstrap=ci_bootstrap)
                lines.append("")
                lines.append(f"**Bootstrap 95% ДИ** (n={ci_bootstrap}):")
                lines.append(f"- tau_ratio (median по сидам): [{lo_t:.2f}x, {hi_t:.2f}x]")
            if sign_accs:
                lo_s, hi_s = bootstrap_ci(sign_accs, n_bootstrap=ci_bootstrap)
                if not tau_ratios:
                    lines.append("")
                    lines.append(f"**Bootstrap 95% ДИ** (n={ci_bootstrap}):")
                lines.append(f"- sign_acc: [{lo_s:.0%}, {hi_s:.0%}]")
        lines.append("")

    # Overall
    lines.append("## Итоговый вердикт")
    lines.append("")
    if "test_A" in data and "test_B" in data:
        va = data["test_A"]["verdict"].get("VERDICT", "—")
        vb = data["test_B"]["verdict"].get("VERDICT", "—")
        if "PASS" in va and "PASS" in vb:
            overall = "**FULL PASS** — IDT G-активность подтверждена по обоим тестам."
        elif "PASS" in va or "PASS" in vb:
            overall = "**PARTIAL PASS** — один из двух тестов прошёл."
        elif "AMBIGUOUS" in va or "AMBIGUOUS" in vb:
            overall = "**AMBIGUOUS** — сигнал есть, не решающий."
        else:
            overall = "**FAIL** — G-активность не подтверждена на данной настройке."
        lines.append(f"- Test A: {va}")
        lines.append(f"- Test B: {vb}")
        lines.append(f"- **Итог:** {overall}")
    else:
        lines.append("Нет данных по одному или обоим тестам.")
    lines.append("")

    text = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description="Построение отчёта из final_report.json")
    ap.add_argument("--input", "-i", type=str, default="final_report.json",
                    help="Путь к final_report.json")
    ap.add_argument("--output", "-o", type=str, default="VALIDATION_REPORT.md",
                    help="Путь к выходному VALIDATION_REPORT.md")
    ap.add_argument("--ci", type=int, default=0,
                    help="Число bootstrap итераций для 95%% ДИ (0 = не считать)")
    args = ap.parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        print(f"Файл не найден: {inp}")
        print("Сначала запустите: python idt_gpu_tests.py --test AB --seeds 10 --out results/")
        return 1
    data = load_report(inp)
    build_report(data, out, ci_bootstrap=args.ci)
    print(f"Отчёт записан: {out}")
    return 0


if __name__ == "__main__":
    exit(main())
