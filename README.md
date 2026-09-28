# Своё Вино — AbobaMakers

Сканер российских вин для кейса РСХБ.Цифра. Демонстрация: https://abobawines.ru/

Приложение распознаёт вино по фотографии, открывает карточку с производителем, регионом, сортом и описанием и подбирает сочетание с блюдом. Первый кандидат показан сразу, альтернативы доступны в раскрываемом списке. Интерфейс адаптирован для телефона и поддерживает установку как PWA.

## Документация

- [Архитектура](ARCHITECTURE.md)
- [Результаты обучения](docs/TRAINING_RESULTS.md)

## Быстрый запуск интерфейса

Python 3.12, PowerShell; команды из корня проекта:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\scripts\run.ps1 -Preview -Port 8081
```

Откройте http://127.0.0.1:8081/. Режим Preview работает без модели; распознавание и подбор блюда в нём недоступны.

## Полноценное распознавание

Веса не поставляются обычным git clone. Необходим согласованный набор:

- `artifacts/base_model/`: `config.json`, `preprocessor_config.json`, `model.safetensors`, `source.json`;
- `artifacts/experiment/`: `index.pt`, `adapter_best.pt`, `manifest.json`;
- `calibration.json` в папке эксперимента — пороги визуального поиска;
- `ocr_index.json` и `hybrid_metrics.json` там же — гибридный поиск и его настройки.

При старте проверяются хеши адаптера и манифеста и ревизия энкодера. Не редактируйте манифест вручную. Артефакты нужно получить вместе с данными проекта либо воспроизвести обучение.

Установка CPU-зависимостей и запуск на Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PAIRING_PROVIDER = 'rules'
.\scripts\run.ps1 -Port 8081
```

## Docker

С подготовленными артефактами, из корня проекта в Linux:

```bash
docker build -t wine-app:local .
docker run -d --name wine-app --restart unless-stopped \
  -p 127.0.0.1:8080:8080 \
  -e PAIRING_PROVIDER=rules \
  -v "$PWD/artifacts:/app/artifacts:ro" \
  --log-opt max-size=10m --log-opt max-file=3 wine-app:local
curl http://127.0.0.1:8080/health
```

Артефакты должны быть доступны для чтения пользователю контейнера UID 10001. Dockerfile использует CPU PyTorch и один процесс API. Для внешнего доступа нужен HTTPS reverse proxy, направляющий запросы на 127.0.0.1:8080. Код и зависимости находятся в образе, модель подключена отдельной папкой. После изменения кода образ пересобирается, контейнер пересоздаётся.

## Сомелье

`PAIRING_PROVIDER=rules` включает локальные правила без внешнего API. Для YandexGPT скопируйте `.env.example` в `.env`, заполните ключ, идентификатор каталога и модель, задайте `PAIRING_PROVIDER=yandex`. Переменные окружения имеют приоритет над файлом. В Docker вместо `-e PAIRING_PROVIDER=rules` используйте `--env-file .env`.

Ключи не публикуются в Git. YandexGPT требует интернета и оплачивается владельцем ключа. Внешней модели передаются блюдо и поля карточки, а не фото. При ошибке YandexGPT автоматического перехода на правила нет. Оценка 0–100 — условная совместимость, не рейтинг вина. Лимиты запросов и расходов приложением пока не реализованы.

## API

| Метод | Назначение |
|---|---|
| GET /health | Готовность распознавателя, размер каталога и pipeline; не проверяет YandexGPT |
| POST /v1/eval/predict | Multipart `image`; ответ `{"slug":"..."}` |
| POST /v1/predict | Multipart `image`; карточка, статус, до пяти кандидатов |
| POST /v1/pairing | JSON `{"slug":"...","dish":"..."}`; рекомендация |

Лимит изображения — 20 МиБ. Оценочный endpoint всегда возвращает ближайший slug. Продуктовый endpoint может рекомендовать переснять фото.

## Обучение и оценка

Ожидаемые пути: фотографии `data/foto`, каталог `data/strapi_output0709.csv`, публичные запросы `data/eval/queries` и `data/eval/queries.tsv`. При другом расположении данных измените аргументы соответствующих скриптов.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_catalog.py --images data/foto --catalog data/strapi_output0709.csv
.\.venv\Scripts\python.exe scripts/download_model.py
.\.venv\Scripts\python.exe -m wine_ml.experiment extract
.\.venv\Scripts\python.exe -m wine_ml.experiment train --epochs 15
.\.venv\Scripts\python.exe scripts/calibrate.py
.\.venv\Scripts\python.exe -m wine_ml.ocr
.\.venv\Scripts\python.exe scripts/evaluate_hybrid.py
```

Исходный эксперимент выполнен на GPU с PyTorch CUDA; для повторения его окружения нужна соответствующая сборка PyTorch. `requirements.lock.txt` фиксирует зависимости эксперимента, Docker использует `requirements.txt`. Загрузчик запрашивает текущую ревизию базовой модели: для точного воспроизведения необходимо сохранить исходные веса и `source.json`.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe scripts/evaluate_public.py
.\.venv\Scripts\python.exe scripts/benchmark.py
```

Последние две команды требуют API на порту 8080 и подготовленных проверочных фото. Для повторного прогона `evaluate_public.py` задайте новый `--output`. Без эталонных ответов скрипт сохраняет предсказания, но не вычисляет точность.

## Веса модели и поисковый индекс

Из-за размера обученные артефакты не хранятся в Git.

Скачать: <https://disk.360.yandex.ru/d/WSXDM0xHz9m0tQ>  
SHA256 архива: `<861EAEF834C9A946B4AAB64D3B2D235B3534A64FC339201C544D28BBDB0DFA87>`

Распакуйте архив в корень репозитория. После распаковки должны
существовать файлы:

- artifacts/base_model/model.safetensors
- artifacts/experiment/index.pt
- artifacts/experiment/adapter_best.pt
- artifacts/experiment/manifest.json

## Ограничения

- Индекс содержит 1970 вин из 2103 уникальных позиций исходного каталога; связи фото и карточек не прошли полную ручную проверку.
- Accuracy@1 93,17% получена на синтетических преобразованиях каталожных фото, а не на независимой полевой выборке.
- Калибровка и API используют разные представления запросов. Гибридный статус основан на согласии рангов CV/OCR и не учитывает визуальный порог; пустой OCR может возвращать нулевые совпадения.
- Отказ для неизвестных вин не откалиброван. Сходство не является вероятностью или F1.
- Распознавания в процессе выполняются последовательно. Показанная задержка не включает передачу фото и очередь.
- Рейтинг Роскачества, определение года урожая, история и избранное не реализованы.
- PWA кэширует интерфейс, но для распознавания и рекомендаций нужен сервер. Для установки на телефон требуется HTTPS. После обновления кэша закройте старые вкладки и откройте приложение снова.
