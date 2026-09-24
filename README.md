# AbobaMakers_Vino

Сканер российских вин для кейса РСХБ.Цифра.

План и объяснение обучения: [docs/PLAN_RU.md](docs/PLAN_RU.md).
Границы компонентов: [ARCHITECTURE.md](ARCHITECTURE.md).
Фактические результаты: [docs/TRAINING_RESULTS.md](docs/TRAINING_RESULTS.md).

## Текущее состояние

Реализованы аудит каталога, извлечение признаков SigLIP 2, обучение остаточного адаптера, OCR-поиск, гибридное ранжирование, локальный API и мобильная веб-страница. Данные: `data/foto`, `data/strapi_output0709.csv`, `data/eval`. Результаты обучения фиксируются в `artifacts/experiment/metrics.json`, расширенные метрики — в `calibration.json` и `hybrid_metrics.json`.

Обучение завершено: 1970 вин в индексе, 15 эпох, выбрана эпоха 13. Синтетический test Accuracy@1: 90.78% → 93.17%. На реальных публичных фото обнаружены несоответствия; эти синтетические метрики нельзя выдавать за точность в магазине.

Это исследовательский прототип: связи по именам предварительные, каталог поиска сокращён до неконфликтующих кандидатов, проверка обучения синтетическая. OCR и мобильный интерфейс реализованы. Отказ откалиброван только для известных вин; полноценная калибровка неизвестных требует размеченных отрицательных фото.

## Аудит

Python 3.10+, сторонние зависимости не нужны. Команды из корня репозитория:

```powershell
python scripts/prepare_catalog.py --images data/foto
python scripts/prepare_catalog.py --images data/foto --catalog data/strapi_output0709.csv
```

Вторая команда требует реального CSV с колонками `Slug` и `Название фото` в UTF-8. Скрипт не изменяет исходные данные. Результаты: `artifacts/data_audit/report.json`, `inventory.csv`, а при наличии каталога — `candidate_links.json` и `content_conflicts.json`. Все связи требуют проверки; статус `candidate` не означает подтверждённую разметку.

## Установка и обучение (Windows, Python 3.12)

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 --timeout 180
.venv/Scripts/python.exe -m pip install -r requirements.lock.txt --timeout 180
.venv/Scripts/python.exe -X utf8 scripts/download_model.py
.venv/Scripts/python.exe -X utf8 scripts/prepare_catalog.py --images data/foto --catalog data/strapi_output0709.csv
.venv/Scripts/python.exe -X utf8 -m wine_ml.experiment extract
.venv/Scripts/python.exe -X utf8 -m wine_ml.experiment train --epochs 15
.venv/Scripts/python.exe -X utf8 scripts/calibrate.py
.venv/Scripts/python.exe -X utf8 -m wine_ml.ocr
.venv/Scripts/python.exe -X utf8 scripts/evaluate_hybrid.py
```

Загрузка зависимостей и официальных весов требует интернета и нескольких гигабайт свободного места. После скачивания энкодер загружается только из `artifacts/base_model`; при распознавании сетевые запросы не выполняются. Переменные окружения и ключи API не требуются. Команды выполняются из корня проекта.

В `artifacts/experiment`: `manifest.json` и `excluded.json` (состав выборки), `features.pt`, `adapter_last.pt` (реально обученные веса последней эпохи), `adapter_best.pt` (выбор по validation, возможно исходное преобразование), `index.pt`, `metrics.json`. При изменении энкодера, обработки фото или каталога признаки и индекс нужно пересоздать.

## Запуск и проверка

```powershell
.\scripts\run.ps1
```

В другом терминале:

```powershell
.venv/Scripts/python.exe -X utf8 scripts/evaluate_public.py
.venv/Scripts/python.exe -X utf8 -m wine_ml.predict data/eval/queries/019c68d0.jpg
.venv/Scripts/python.exe -X utf8 -m unittest discover -s tests
```

Откройте `http://127.0.0.1:8080/`. `GET /health` — готовность и размер каталога. `POST /v1/eval/predict` — поле `image`, ответ `{"slug":"..."}`. `POST /v1/predict` — карточка, OCR и диагностические кандидаты. `POST /v1/pairing` — рекомендация к блюду. Оценочный endpoint всегда возвращает ближайший slug для совместимости; продуктовый API умеет вернуть `low_confidence_reshoot`.

`scripts/evaluate_public.py` — собственный последовательный HTTP-прогон для Windows с полями JSONL из скрипта организатора; он не подменяет официальный Bash-скрипт. Существующий файл результатов не перезаписывается: задайте другой `--output` для повторного прогона. Публичные ответы закрыты, поэтому точность по этим трём фото не рассчитывается.

Отдельная диагностическая проверка локальных деталей (не используется основным API):

```powershell
.venv/Scripts/python.exe -X utf8 -m wine_ml.local_features
.venv/Scripts/python.exe -X utf8 -m wine_ml.check_geometry
```

Она сохраняет `local_features.npz` и `artifacts/geometry_ablation.json`. На просмотренных публичных фото геометрия не решила путаницу похожих серий; увеличение числа совпавших элементов оформления не доказывает совпадение вина.
