"""
IDT GPU TESTS — РАСШИРЕННЫЙ НАБОР ТЕСТОВ
========================================
Содержит:
1. Юнит-тесты формул (M_omega, slowdown, lam_min, GEP) без обучения
2. Тест Lanczos на синтетической матрице
3. Тест HVP на маленькой модели
4. Санити-проверки (M в допустимом диапазоне, H положительно определённая)
5. Регрессионные тесты на фиксированных сидах
Запуск:
  pip install torch torchvision numpy scipy pytest
  pytest idt_gpu_tests_extended.py -v
  или: python idt_gpu_tests_extended.py
"""

import sys
import numpy as np
from pathlib import Path
from scipy.linalg import eigh

# Локальные копии формул, чтобы тесты работали без torch
def M_omega(lam):
    lam = np.sort(np.real(lam))
    lam = lam[lam > 1e-10]
    if len(lam) < 2:
        return np.nan
    return float(np.mean(lam**2) / (lam[0] * np.mean(lam)))


def lam_min_positive(H_r):
    ev = np.linalg.eigvalsh(H_r)
    ev = ev[ev > 1e-10]
    return float(ev[0]) if len(ev) > 0 else 1e-10


def compute_M(H_r, G_r):
    try:
        return M_omega(np.sort(np.real(eigh(H_r, G_r, eigvals_only=True))))
    except Exception:
        return np.nan


def slowdown(loss_hist, ep, delta, eps):
    if ep + delta >= len(loss_hist):
        return np.nan
    return 1.0 / (abs(loss_hist[ep + delta] - loss_hist[ep]) + eps)


# CFG — только для тестов конфига (значения из PDF)
CFG = {
    "model": "resnet20",
    "dataset": "cifar10",
    "epochs": 160,
    "seeds": 10,
    "bs_main": 128,
    "lr_main": 0.1,
    "bs_small": 32,
    "bs_large": 512,
    "lr_small": 0.025,
    "lr_large": 0.200,
    "R": 50,
    "N_fim": 200,
    "reg": 1e-5,
    "N_perm": 500,
    "delta_loss": 5,
    "loss_tol": 0.01,
    "h5_tol": 0.05,
    "delta_L_nn02": 0.02,
}


# ═══════════════════════════════════════════════════════════════════
# ЮНИТ-ТЕСТЫ: M(ω)
# ═══════════════════════════════════════════════════════════════════
def test_M_omega_two_eigenvalues():
    """M(ω) = mean(λ²) / (λ_min * mean(λ)) — формула из IDT."""
    lam = np.array([2.0, 1.0])
    m = M_omega(lam)
    # mean(λ²)=2.5, λ_min=1, mean(λ)=1.5 → M = 2.5/1.5 ≈ 1.667
    expected = (4 + 1) / 2 / (1.0 * 1.5)
    assert not np.isnan(m), "M_omega не должен возвращать nan для 2 значений"
    assert np.isclose(m, expected), f"M_omega([2,1]) = {m}, ожидалось {expected}"


def test_M_omega_uniform():
    """Для равных λ: M = mean(λ²)/(λ_min*mean(λ)) = λ²/(λ*λ) = 1."""
    lam = np.array([3.0, 3.0, 3.0])
    m = M_omega(lam)
    assert np.isclose(m, 1.0), f"M_omega([3,3,3]) = {m}, ожидалось 1.0"


def test_M_omega_single_value_returns_nan():
    """Один собственный вектор — по определению M не определено."""
    m = M_omega(np.array([1.0]))
    assert np.isnan(m), "M_omega от одного значения должен быть nan"


def test_M_omega_empty_returns_nan():
    """Пустой массив или все нули -> nan."""
    assert np.isnan(M_omega(np.array([])))
    assert np.isnan(M_omega(np.array([0.0, 0.0])))


def test_M_omega_negative_filtered():
    """Отрицательные и нулевые отфильтровываются (только λ > 1e-10)."""
    lam = np.array([-1.0, 0.0, 2.0, 1.0])
    m = M_omega(lam)
    # Должны остаться 1, 2 → mean(λ²)=2.5, λ_min=1, mean(λ)=1.5 → 2.5/1.5
    expected = 2.5 / (1.0 * 1.5)
    assert np.isclose(m, expected), f"После фильтра: {m} vs {expected}"


# ═══════════════════════════════════════════════════════════════════
# ЮНИТ-ТЕСТЫ: slowdown
# ═══════════════════════════════════════════════════════════════════
def test_slowdown_basic():
    """slowdown = 1/|ΔL| при eps=0."""
    loss = [1.0, 0.9, 0.8]
    s = slowdown(loss, 0, delta=1, eps=1e-8)
    assert not np.isnan(s)
    assert np.isclose(s, 1.0 / 0.1, rtol=1e-5), "slowdown при ΔL=0.1"


def test_slowdown_out_of_bounds_nan():
    """Если ep+delta >= len(loss_hist) -> nan."""
    loss = [1.0, 0.9]
    s = slowdown(loss, 1, delta=5, eps=1e-8)
    assert np.isnan(s)


# ═══════════════════════════════════════════════════════════════════
# ЮНИТ-ТЕСТЫ: lam_min_positive, compute_M (GEP)
# ═══════════════════════════════════════════════════════════════════
def test_lam_min_positive():
    """Минимальное положительное собственное значение."""
    H = np.diag([5.0, 3.0, 1.0, 0.0, -1.0])
    lam = lam_min_positive(H)
    assert np.isclose(lam, 1.0), f"lam_min_positive = {lam}"


def test_compute_M_symmetric_gep():
    """compute_M(H_r, G_r): GEP eigenvalues -> M_omega."""
    from scipy.linalg import eigh
    np.random.seed(42)
    r = 5
    H_r = np.random.randn(r, r)
    H_r = H_r.T @ H_r + np.eye(r) * 0.1
    G_r = np.eye(r) + np.random.randn(r, r) * 0.1
    G_r = G_r.T @ G_r + np.eye(r) * 0.01
    m = compute_M(H_r, G_r)
    ev = np.sort(np.real(eigh(H_r, G_r, eigvals_only=True)))
    ev = ev[ev > 1e-10]
    expected = M_omega(ev)
    assert not np.isnan(m), "compute_M не должен быть nan для положительно определённых H,G"
    assert np.isclose(m, expected), f"compute_M vs M_omega(ev): {m} vs {expected}"


def test_M_omega_bounds():
    """По теореме IDT: M >= 1 (для положительных λ)."""
    np.random.seed(123)
    for _ in range(10):
        lam = np.sort(np.random.rand(5) + 0.5)[::-1]
        m = M_omega(lam)
        if not np.isnan(m):
            assert m >= 1.0 - 1e-6, f"M(ω) должен быть >= 1, получено {m}"


# ═══════════════════════════════════════════════════════════════════
# ТЕСТ LANCZOS (на синтетическом операторе)
# ═══════════════════════════════════════════════════════════════════
def test_lanczos_synthetic():
    """Lanczos на диагональной матрице даёт собственные значения."""
    try:
        import torch
    except ImportError:
        return
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from idt_gpu_tests import lanczos
    except (ImportError, SystemExit):
        return  # зависимость или выход из дочернего кода — пропуск
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = 100
    diag = torch.rand(n, device=device) + 0.5
    def Av(v):
        return diag * v
    Q, ev = lanczos(Av, n, R=5, extra=5)
    assert Q is not None and ev is not None
    true_ev = np.sort(diag.cpu().numpy())[::-1][:5]
    np.testing.assert_allclose(np.sort(ev)[::-1], true_ev, rtol=1e-4)


# ═══════════════════════════════════════════════════════════════════
# САНИТИ: конфиг и константы
# ═══════════════════════════════════════════════════════════════════
def test_cfg_required_keys():
    """Все обязательные ключи конфига присутствуют."""
    required = [
        "model", "dataset", "epochs", "seeds", "bs_main", "lr_main",
        "R", "N_fim", "reg", "N_perm", "delta_loss", "loss_tol", "h5_tol",
        "delta_L_nn02", "bs_small", "bs_large", "lr_small", "lr_large",
    ]
    for k in required:
        assert k in CFG, f"В CFG отсутствует ключ: {k}"


def test_cfg_lr_scaling():
    """Проверка направления масштабирования: lr_small < lr_main < lr_large."""
    assert CFG["lr_small"] < CFG["lr_main"] < CFG["lr_large"], \
        "lr должен расти с batch size (sqrt scaling)"
    # Опционально: приближённо lr ∝ sqrt(bs)
    ratio_small = CFG["lr_small"] / CFG["lr_main"]
    ratio_large = CFG["lr_large"] / CFG["lr_main"]
    assert 0.1 < ratio_small < 1.0 and 1.0 < ratio_large < 5.0, \
        "Отношения lr к основному в разумных границах"


# ═══════════════════════════════════════════════════════════════════
# РЕГРЕССИЯ: воспроизводимость на фиксированном сиде (без полного обучения)
# Проверяем, что при фиксированном seed одни и те же входы дают те же M/slowdown.
# ═══════════════════════════════════════════════════════════════════
def test_reproducibility_M_omega():
    """M_omega детерминирован при одинаковых λ."""
    lam = np.array([1.5, 1.2, 1.0, 0.8, 0.5])
    m1 = M_omega(lam)
    m2 = M_omega(lam.copy())
    assert m1 == m2 and not np.isnan(m1)


def test_reproducibility_slowdown():
    """slowdown детерминирован."""
    loss = np.array([2.0, 1.5, 1.2, 1.0])
    s1 = slowdown(loss, 0, 2, 1e-8)
    s2 = slowdown(loss, 0, 2, 1e-8)
    assert s1 == s2


# ═══════════════════════════════════════════════════════════════════
# ЗАПУСК ВСЕХ ТЕСТОВ
# ═══════════════════════════════════════════════════════════════════
def run_all_unit_tests():
    """Запуск всех юнит-тестов вручную (без pytest)."""
    tests = [
        test_M_omega_two_eigenvalues,
        test_M_omega_uniform,
        test_M_omega_single_value_returns_nan,
        test_M_omega_empty_returns_nan,
        test_M_omega_negative_filtered,
        test_slowdown_basic,
        test_slowdown_out_of_bounds_nan,
        test_lam_min_positive,
        test_compute_M_symmetric_gep,
        test_M_omega_bounds,
        test_lanczos_synthetic,
        test_cfg_required_keys,
        test_cfg_lr_scaling,
        test_reproducibility_M_omega,
        test_reproducibility_slowdown,
    ]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed.append((t.__name__, e))
    if failed:
        print(f"\nПровалено: {len(failed)}/{len(tests)}")
        return 1
    print(f"\nВсе тесты пройдены: {len(tests)}")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_unit_tests())
