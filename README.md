# Video Transcriber

Сервис для транскрибации длинных видео (1-5 часов) на базе `faster-whisper` + `Modal` (GPU T4), с веб-интерфейсом и прогрессом через SSE.

## Что такое Modal

`Modal` — это облачная платформа для запуска Python-кода в контейнерах, включая GPU-нагрузки, без ручной настройки серверов. При регистрации даётся бесплатно баланс в 5$ на серверные ресурсы.
В этом проекте Modal используется для:

- деплоя веб-приложения,
- запуска GPU-воркера для транскрибации,
- хранения файлов и статусов задач в `modal.Volume`.

## Возможности

- Загрузка локального видео/аудио файла
- Транскрибация по YouTube-ссылке (`yt-dlp`)
- Результат в Markdown с таймкодами
- Мультиязычность: `auto`, `ru`, `en`, `uk`, `de`, `fr`, `es`, `zh`, `ja`, `ko`
- Отслеживание статуса задачи в реальном времени

## Требования

- Python 3.11+
- Аккаунт в Modal
- Установленный Modal CLI
- Пройденная аутентификация в Modal (`modal setup`)

## Быстрый старт

### 1. Установка CLI и авторизация

```bash
pip install -r requirements.txt
modal setup
```

`modal setup` обязателен: команда открывает браузер для входа в аккаунт и сохраняет токен доступа для CLI.

Windows-эквивалент:

```powershell
py -m pip install -r requirements.txt
py -m modal setup
```

### 2. Локальная разработка

```bash
modal serve app.py
```

Windows:

```powershell
py -m modal serve app.py
```

### 3. Деплой

```bash
modal deploy app.py
```

Windows:

```powershell
py -m modal deploy app.py
```

## Как получить ссылку на приложение после deploy

После `modal deploy app.py` Modal выводит URL веб-приложения в консоль.
Обычно формат такой:

`https://<workspace>--video-transcriber-web.modal.run`

Если ссылка не отобразилась в выводе, можно посмотреть активные приложения:

```bash
modal app list
```

Windows:

```powershell
py -m modal app list
```

## Формат результата

Результат транскрибации возвращается в формате **Markdown (`.md`)**.
Файл содержит:

- метаданные (дата, язык, модель),
- текст с таймкодами,
- блочную структуру по времени.

Скачивание результата: `/result/{job_id}`.

## Конфигурация (ENV)

| Переменная | По умолчанию | Назначение |
|---|---:|---|
| `WHISPER_MODEL` | `large-v3` | Модель Whisper |
| `WHISPER_COMPUTE_TYPE` | `float16` | Тип вычислений на GPU |
| `WHISPER_BATCH_SIZE` | `16` | Размер батча для batched inference |
| `WHISPER_BEAM_SIZE` | `5` | Beam size (качество/скорость) |
| `VAD_MIN_SILENCE_MS` | `500` | Минимальная тишина для VAD |
| `MAX_UPLOAD_BYTES` | `4294967296` | Лимит загрузки (4 ГБ, лимит Modal web endpoint) |
| `UPLOAD_CHUNK_BYTES` | `4194304` | Размер чанка загрузки |
| `STATUS_POLL_SECONDS` | `2.0` | Интервал опроса статуса |
| `SSE_HEARTBEAT_SECONDS` | `15.0` | SSE heartbeat |
| `GPU_TIMEOUT_SECONDS` | `14400` | Таймаут GPU-транскрибации |
| `YOUTUBE_TIMEOUT_SECONDS` | `3600` | Таймаут YouTube-скачивания |

## Как устроено

1. Клиент отправляет файл в `/upload` или ссылку в `/youtube`
2. Статус и файлы задачи сохраняются в `modal.Volume`
3. GPU-класс `Transcriber` выполняет распознавание
4. Результат сохраняется как `result.md`
5. Фронтенд читает прогресс через `/status/{job_id}` и скачивает `/result/{job_id}`

## Структура проекта

```text
video-transcriber/
├── app.py
├── static/
│   └── index.html
├── requirements.txt
├── .gitignore
└── README.md
```
