#!/bin/bash
# Полный прогон IDT GPU тестов и генерация отчёта с таблицей метрик и вердиктом.
# Требуется: pip install torch torchvision numpy scipy
# Время: ~8–14 ч на одном GPU для --seeds 10.
#
# Использование:
#   ./run_full_validation.sh                  # по умолчанию: 10 seeds, папка results/
#   ./run_full_validation.sh --seeds 3        # быстрее: 3 seeds
#   ./run_full_validation.sh --quick          # проверка кода: 1 seed, 20 эпох
#   ./run_full_validation.sh --out my_run --ci 2000   # папка my_run, bootstrap ДИ 2000

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
OUT_DIR="results"
SEEDS="10"
QUICK=""
CI="0"
while [[ $# -gt 0 ]]; do
  case $1 in
    --out)   OUT_DIR="$2"; shift 2 ;;
    --seeds) SEEDS="$2";   shift 2 ;;
    --quick) QUICK="--quick"; shift ;;
    --ci)    CI="$2";      shift 2 ;;
    *)       shift ;;
  esac
done

echo "=== IDT GPU Full Validation ==="
echo "Output: $OUT_DIR  Seeds: $SEEDS  Bootstrap CI: $CI"
echo ""

EXTRA=""
if [[ -n "$QUICK" ]]; then
  EXTRA="--quick"
  echo "Режим: QUICK (1 seed, 20 эпох)"
else
  echo "Режим: полный (A+B, $SEEDS seeds)"
fi
echo ""

python idt_gpu_tests.py --test AB --seeds "$SEEDS" --out "$OUT_DIR" $EXTRA

if [[ -f "$OUT_DIR/final_report.json" ]]; then
  CI_ARG=""
  [[ "$CI" != "0" ]] && CI_ARG="--ci $CI"
  python build_report_from_results.py --input "$OUT_DIR/final_report.json" \
    --output "$OUT_DIR/VALIDATION_REPORT.md" $CI_ARG
  echo ""
  echo "Отчёт с таблицей и вердиктом: $OUT_DIR/VALIDATION_REPORT.md"
else
  echo "Файл $OUT_DIR/final_report.json не создан — отчёт не сгенерирован."
fi
