# Шаблон отчёта валидации IDT GPU Tests

После запуска тестов заполните этот файл данными из `results/VALIDATION_REPORT.md` (или переименуйте тот файл в этот и положите в корень репозитория для включения в доказательную базу).

---

## Как получить отчёт

1. **Установка (один раз)**  
   ```bash
   pip install torch torchvision numpy scipy
   ```

2. **Полный прогон (рекомендуется)**  
   ```bash
   cd "IDT TEST"
   chmod +x run_full_validation.sh
   ./run_full_validation.sh --out results
   ```  
   Время: ~8–14 ч на одном GPU (RTX 3090 / A100). Результаты: `results/final_report.json`, `results/VALIDATION_REPORT.md`.

3. **Быстрая проверка (без полноценной валидации)**  
   ```bash
   python idt_gpu_tests.py --test AB --seeds 1 --quick --out quick_run
   python build_report_from_results.py --input quick_run/final_report.json --output quick_run/VALIDATION_REPORT.md
   ```

4. **Отчёт с bootstrap 95% ДИ**  
   ```bash
   ./run_full_validation.sh --out results --ci 2000
   ```
   или после прогона:
   ```bash
   python build_report_from_results.py --input results/final_report.json --output results/VALIDATION_REPORT.md --ci 2000
   ```

---

## Место для таблицы и вердикта (заполнить после прогона)

После выполнения шагов выше скопируйте сюда содержимое `results/VALIDATION_REPORT.md` или вставьте таблицы и вердикт вручную.

### Test A — IDT-CIFAR-01

| Seed | T | ρ_H | ρ_M | perm_mean | pctile_M | beats_H | >p95 |
|------|---|-----|-----|-----------|----------|--------|-----|
| …    | … | …   | …   | …         | …        | …      | …   |

**Критерии:** C1 … C4  
**Вердикт A:** …

### Test B — IDT-NN-02

| Seed | n_matched | tau_ratio | M_ratio | MH_ratio | sign_acc | ρ_M | PASS |
|------|-----------+-----------+---------|----------|----------|-----|------|
| …    | …         | …         | …       | …        | …        | …   | …    |

**Критерии:** …  
**Вердикт B:** …

### Итоговый вердикт

- Test A: …
- Test B: …
- **Итог:** FULL PASS / PARTIAL PASS / AMBIGUOUS / FAIL

---

---

## Опционально: усиление выводов

- **Доверительные интервалы:** при полном прогоне задайте `--ci 2000` в `run_full_validation.sh` или вызовите `build_report_from_results.py --ci 2000` — в отчёт добавятся bootstrap 95% ДИ для pctile_M, Δρ, tau_ratio, sign_acc.
- **Несколько архитектур/датасетов:** текущий код заточен под ResNet-20 и CIFAR-10. Добавление других моделей (например, MLP, VGG) или датасетов потребует правок в `idt_gpu_tests.py` (выбор модели/датасета по аргументу или конфигу) и повторных прогонов с сохранением отчётов по каждому сценарию.

---

*Для полной доказательной базы по IDT в нейросетях необходимо сохранить этот отчёт с реальными данными в репозитории или приложить к публикации.*
