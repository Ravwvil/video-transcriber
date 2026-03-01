"""Video Transcriber on Modal.

Transcribes long videos (1-5 hours) using faster-whisper.
Web UI supports local uploads and YouTube URLs with SSE progress.
"""

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import modal


# Configuration
MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3")
GPU_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
BUILD_COMPUTE_TYPE = "int8"

WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "16"))
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "500"))

GPU_TIMEOUT_SECONDS = int(os.getenv("GPU_TIMEOUT_SECONDS", str(4 * 3600)))
YOUTUBE_TIMEOUT_SECONDS = int(os.getenv("YOUTUBE_TIMEOUT_SECONDS", "3600"))

MODAL_WEB_BODY_LIMIT_BYTES = 4 * 1024 * 1024 * 1024  # Modal web endpoint limit
MAX_UPLOAD_BYTES = min(
    int(os.getenv("MAX_UPLOAD_BYTES", str(MODAL_WEB_BODY_LIMIT_BYTES))),
    MODAL_WEB_BODY_LIMIT_BYTES,
)
MAX_UPLOAD_MB = MAX_UPLOAD_BYTES // (1024 * 1024)
UPLOAD_CHUNK_BYTES = int(os.getenv("UPLOAD_CHUNK_BYTES", str(4 * 1024 * 1024)))
STATUS_POLL_SECONDS = float(os.getenv("STATUS_POLL_SECONDS", "2.0"))
SSE_HEARTBEAT_SECONDS = float(os.getenv("SSE_HEARTBEAT_SECONDS", "15.0"))

VOLUME_PATH = "/data"
JOBS_DIR = Path(VOLUME_PATH) / "jobs"

CUDA_BASE_IMAGE = "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04"
CUDA_LIBRARY_PATH = (
    "/usr/local/cuda/lib64:"
    "/usr/local/nvidia/lib64:"
    "/usr/lib/x86_64-linux-gnu:"
    "/usr/lib"
)

YOUTUBE_URL_RE = re.compile(r"(youtube\.com|youtu\.be)", re.IGNORECASE)

SUPPORTED_LANGUAGES = {
    "auto",
    "de",
    "en",
    "es",
    "fr",
    "ja",
    "ko",
    "ru",
    "uk",
    "zh",
}


# Modal app and infrastructure
app = modal.App("video-transcriber")
volume = modal.Volume.from_name("video-transcriber-data", create_if_missing=True)


def download_model() -> None:
    """Pre-download Whisper model during image build (cached in layer)."""
    from faster_whisper import WhisperModel

    WhisperModel(MODEL_NAME, device="cpu", compute_type=BUILD_COMPUTE_TYPE)


image = (
    modal.Image.from_registry(
        CUDA_BASE_IMAGE,
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install("ffmpeg")
    .pip_install(
        "faster-whisper>=1.0.0",
        "yt-dlp",
        "fastapi[standard]",
        "python-multipart",
    )
    .env(
        {
            "HF_HOME": "/models",
            "PYTHONUNBUFFERED": "1",
            "LD_LIBRARY_PATH": CUDA_LIBRARY_PATH,
        }
    )
    .run_function(download_model)
    .add_local_dir("static", remote_path="/app/static")
)


# Helpers
def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _ts(seconds: float) -> str:
    """Seconds to HH:MM:SS."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _dur(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, _ = divmod(rem, 60)
    return f"{h}ч {m}мин" if h else f"{m}мин"


def _safe_filename(name: str) -> str:
    """Sanitize user-supplied filename for the filesystem."""
    name = os.path.basename(name)
    return re.sub(r"[^\w\-.]", "_", name) or "upload"


def _normalize_language(language: str | None) -> str:
    lang = (language or "auto").strip().lower()
    return lang if lang in SUPPORTED_LANGUAGES else "auto"


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp_path.open("w", encoding=encoding) as f:
        f.write(content)
    os.replace(tmp_path, path)


def _read_status(job_id: str) -> dict[str, Any] | None:
    status_path = _job_dir(job_id) / "status.json"
    if not status_path.exists():
        return None

    for _ in range(3):
        try:
            with status_path.open(encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            time.sleep(0.05)
    return None


def _update(
    job_id: str,
    status: str,
    progress: float = 0.0,
    message: str = "",
    result_file: str = "",
) -> None:
    payload = {
        "status": status,
        "progress": round(max(0.0, min(1.0, progress)), 3),
        "message": message,
        "result_file": result_file,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_text(
        _job_dir(job_id) / "status.json",
        json.dumps(payload, ensure_ascii=False),
    )


def _run_subprocess(
    cmd: list[str],
    timeout: int,
    retries: int = 0,
    retry_backoff_seconds: float = 2.0,
) -> str:
    """Run command with retry and return stripped stdout."""
    import subprocess

    last_err = ""
    for attempt in range(retries + 1):
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            last_err = f"timeout after {timeout}s: {exc}"
        else:
            if proc.returncode == 0:
                return proc.stdout.strip()
            last_err = (proc.stderr or proc.stdout or "").strip()

        if attempt < retries:
            time.sleep(retry_backoff_seconds * (attempt + 1))

    raise RuntimeError(f"Command failed: {cmd[0]} ({last_err[:800]})")


def _markdown_header(filename: str, duration: float, language: str) -> str:
    lines = [
        f"# Транскрипция: {filename}",
        "",
        f"- **Дата:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- **Длительность:** {_dur(duration)}",
        f"- **Язык:** {language}",
        f"- **Модель:** Whisper {MODEL_NAME} ({GPU_COMPUTE_TYPE})",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def _write_markdown_body(
    result_path: Path,
    segments: Iterable[Any],
    duration: float,
    job_id: str,
) -> int:
    block_size = 300  # 5 minutes
    block_start = 0
    block_end = block_size
    block_header_written = False
    segment_count = 0
    last_report = time.monotonic()

    with result_path.open("a", encoding="utf-8") as f:
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue

            while seg.start >= block_end:
                block_start = block_end
                block_end += block_size
                block_header_written = False

            if not block_header_written:
                f.write(f"## {_ts(block_start)} - {_ts(block_end)}\n\n")
                block_header_written = True

            f.write(f"**[{_ts(seg.start)}]** {text}\n\n")
            segment_count += 1

            if duration > 0:
                now = time.monotonic()
                if segment_count % 25 == 0 or (now - last_report) >= 5:
                    pct = min(0.18 + 0.72 * (seg.end / duration), 0.90)
                    _update(
                        job_id,
                        "processing",
                        pct,
                        f"Транскрибация: {_ts(seg.end)} / {_dur(duration)}",
                    )
                    volume.commit()
                    last_report = now

    return segment_count


# GPU transcription worker
@app.cls(
    image=image,
    gpu="T4",
    timeout=GPU_TIMEOUT_SECONDS,
    volumes={VOLUME_PATH: volume},
    scaledown_window=300,
)
class Transcriber:
    """Keeps model loaded between calls (container reuse)."""

    @modal.enter()
    def load_model(self) -> None:
        import ctypes
        import glob

        from faster_whisper import BatchedInferencePipeline, WhisperModel

        try:
            ctypes.CDLL("libcublas.so.12")
        except OSError as exc:
            discovered = sorted(
                set(
                    glob.glob("/usr/local/cuda/**/libcublas.so*", recursive=True)
                    + glob.glob("/usr/local/nvidia/**/libcublas.so*", recursive=True)
                    + glob.glob("/usr/lib/**/libcublas.so*", recursive=True)
                    + glob.glob(
                        "/usr/local/lib/python3.11/site-packages/**/libcublas.so*",
                        recursive=True,
                    )
                )
            )
            raise RuntimeError(
                "CUDA library check failed for libcublas.so.12. "
                f"LD_LIBRARY_PATH={os.getenv('LD_LIBRARY_PATH', '')}. "
                f"Discovered candidates={discovered[:8]}. Original error: {exc}"
            ) from exc

        self.model = WhisperModel(
            MODEL_NAME,
            device="cuda",
            compute_type=GPU_COMPUTE_TYPE,
        )
        self.pipeline = BatchedInferencePipeline(model=self.model)

    @modal.method()
    def transcribe(self, job_id: str, file_path: str, language: str = "auto") -> str:
        volume.reload()
        job_dir = _job_dir(job_id)
        input_path = Path(file_path)
        result_path = job_dir / "result.md"

        try:
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")

            _update(job_id, "processing", 0.05, "Подготовка транскрибации…")
            volume.commit()

            lang = _normalize_language(language)
            if lang == "auto":
                lang = None

            segments_gen, info = self.pipeline.transcribe(
                str(input_path),
                language=lang,
                batch_size=WHISPER_BATCH_SIZE,
                beam_size=WHISPER_BEAM_SIZE,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": VAD_MIN_SILENCE_MS},
                condition_on_previous_text=False,
            )
            duration = float(info.duration or 0.0)

            _atomic_write_text(
                result_path,
                _markdown_header(
                    input_path.name,
                    duration,
                    info.language or "auto",
                ),
            )

            _update(job_id, "processing", 0.18, f"Транскрибация ({_dur(duration)})…")
            volume.commit()

            segment_count = _write_markdown_body(
                result_path,
                segments_gen,
                duration,
                job_id,
            )

            if segment_count == 0:
                with result_path.open("a", encoding="utf-8") as f:
                    f.write("_Речь не обнаружена._\n")

            if input_path.exists() and _is_under(input_path, job_dir):
                input_path.unlink()

            _update(job_id, "completed", 1.0, "Готово!", str(result_path))
            volume.commit()
            return str(result_path)

        except Exception as exc:
            _update(job_id, "error", 0, f"Ошибка: {exc}")
            volume.commit()
            raise


# YouTube download -> transcribe
@app.function(
    image=image,
    timeout=YOUTUBE_TIMEOUT_SECONDS,
    volumes={VOLUME_PATH: volume},
)
def download_and_transcribe_youtube(
    job_id: str,
    url: str,
    language: str = "auto",
) -> None:
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        _update(job_id, "downloading", 0.02, "Скачивание с YouTube…")
        volume.commit()

        out_tpl = str(job_dir / "%(title).80s.%(ext)s")
        stdout = _run_subprocess(
            [
                "yt-dlp",
                "--no-playlist",
                "--restrict-filenames",
                "--retries",
                "10",
                "--fragment-retries",
                "10",
                "--socket-timeout",
                "30",
                "--no-warnings",
                "-f",
                "bestaudio[ext=m4a]/bestaudio/best",
                "-o",
                out_tpl,
                "--print",
                "after_move:filepath",
                url,
            ],
            timeout=YOUTUBE_TIMEOUT_SECONDS,
            retries=2,
        )

        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("yt-dlp не вернул путь к скачанному файлу")
        file_path = lines[-1]
        if not file_path or not Path(file_path).exists():
            raise RuntimeError("yt-dlp не вернул путь к скачанному файлу")

        _update(job_id, "queued", 0.10, "Скачано, запуск транскрибации…")
        volume.commit()

        Transcriber().transcribe.spawn(job_id, file_path, _normalize_language(language))

    except Exception as exc:
        _update(job_id, "error", 0, f"Ошибка загрузки: {exc}")
        volume.commit()


# Web UI (FastAPI ASGI)
@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    scaledown_window=300,
)
@modal.concurrent(max_inputs=40)
@modal.asgi_app()
def web():
    import asyncio

    from fastapi import FastAPI, File, Form, Request, UploadFile
    from fastapi.responses import (
        FileResponse,
        HTMLResponse,
        JSONResponse,
        StreamingResponse,
    )

    api = FastAPI(title="Video Transcriber")

    @api.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "ok": True,
                "model": MODEL_NAME,
                "compute_type": GPU_COMPUTE_TYPE,
                "max_upload_bytes": MAX_UPLOAD_BYTES,
            }
        )

    @api.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        with open("/app/static/index.html", encoding="utf-8") as f:
            return HTMLResponse(f.read())

    @api.post("/upload")
    async def upload_video(
        request: Request,
        file: UploadFile = File(...),
        language: str = Form("auto"),
    ) -> JSONResponse:
        job_id = uuid.uuid4().hex[:8]
        job_dir = _job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        safe_name = _safe_filename(file.filename or "upload")
        file_path = job_dir / safe_name
        total_bytes = 0

        _update(job_id, "uploading", 0.01, "Загрузка файла…")
        await volume.commit.aio()

        try:
            content_length = request.headers.get("content-length")
            if content_length and content_length.isdigit():
                # Content-Length includes multipart overhead, so keep a small buffer.
                if int(content_length) > (MAX_UPLOAD_BYTES + 1_000_000):
                    raise ValueError(
                        "Файл слишком большой для web upload в Modal. "
                        f"Лимит: {MAX_UPLOAD_MB} MB"
                    )

            with file_path.open("wb") as f:
                while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_UPLOAD_BYTES:
                        raise ValueError(
                            "Файл слишком большой. "
                            f"Лимит: {MAX_UPLOAD_MB} MB"
                        )
                    f.write(chunk)

            if total_bytes == 0:
                raise ValueError("Пустой файл")

            _update(job_id, "queued", 0.05, "Файл загружен, ожидание GPU…")
            await volume.commit.aio()

            Transcriber().transcribe.spawn(
                job_id,
                str(file_path),
                _normalize_language(language),
            )
            return JSONResponse({"job_id": job_id})

        except ValueError as exc:
            if file_path.exists():
                file_path.unlink()
            _update(job_id, "error", 0, f"Ошибка: {exc}")
            await volume.commit.aio()
            return JSONResponse({"error": str(exc)}, status_code=413)
        except Exception as exc:
            if file_path.exists():
                file_path.unlink()
            _update(job_id, "error", 0, f"Ошибка: {exc}")
            await volume.commit.aio()
            return JSONResponse({"error": str(exc)}, status_code=500)
        finally:
            await file.close()

    @api.post("/youtube")
    async def youtube(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Некорректный JSON"}, status_code=400)

        url = (body.get("url") or "").strip()
        language = _normalize_language(body.get("language"))

        if not url:
            return JSONResponse({"error": "URL обязателен"}, status_code=400)
        if not YOUTUBE_URL_RE.search(url):
            return JSONResponse(
                {"error": "Поддерживаются только YouTube URL"},
                status_code=400,
            )

        job_id = uuid.uuid4().hex[:8]
        _job_dir(job_id).mkdir(parents=True, exist_ok=True)

        _update(job_id, "queued", 0.0, "В очереди…")
        await volume.commit.aio()

        download_and_transcribe_youtube.spawn(job_id, url, language)
        return JSONResponse({"job_id": job_id})

    @api.get("/status/{job_id}")
    async def status_sse(job_id: str, request: Request) -> StreamingResponse:
        async def stream():
            last_heartbeat = time.monotonic()
            while True:
                if await request.is_disconnected():
                    break

                await volume.reload.aio()
                payload = _read_status(job_id) or {
                    "status": "queued",
                    "progress": 0,
                    "message": "В очереди…",
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                if payload.get("status") in ("completed", "error"):
                    break

                now = time.monotonic()
                if now - last_heartbeat >= SSE_HEARTBEAT_SECONDS:
                    yield ": ping\n\n"
                    last_heartbeat = now

                await asyncio.sleep(STATUS_POLL_SECONDS)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @api.get("/result/{job_id}")
    async def result(job_id: str):
        await volume.reload.aio()
        path = _job_dir(job_id) / "result.md"
        if not path.exists():
            return JSONResponse({"error": "Результат не найден"}, status_code=404)
        return FileResponse(
            str(path),
            media_type="text/markdown",
            filename=f"transcription-{job_id}.md",
        )

    return api
