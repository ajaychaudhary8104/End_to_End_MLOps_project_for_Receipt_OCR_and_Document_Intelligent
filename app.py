from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from tempfile import TemporaryDirectory
from datetime import datetime, timezone
import asyncio
import json
import os
import shutil
import time
import uuid

import uvicorn

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Request,
)
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from prometheus_fastapi_instrumentator import Instrumentator

from src.receipt_intelligence.config.configuration import ConfigurationManager
from src.receipt_intelligence.components.inference import ReceiptInferencePipeline


from src.receipt_intelligence import logger


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

TEMPLATE_DIR = ROOT_DIR / "templates"
STATIC_DIR = ROOT_DIR / "static"

INDEX_HTML = TEMPLATE_DIR / "index.html"

PROJECT_ROOT = ROOT_DIR.parent

MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/webp",
}


# ============================================================
# REQUEST MODELS
# ============================================================

class InferenceHealthResponse(BaseModel):

    model_config = ConfigDict(
        protected_namespaces=()
    )

    status: str
    model_loaded: bool
    version: str
    pipeline_ready: bool
    timestamp_utc: str


class BatchInferenceRequest(BaseModel):

    images: List[str] = Field(
        ...,
        min_length=1
    )


# ============================================================
# RESPONSE MODELS
# ============================================================

class PredictionResponse(BaseModel):

    success: bool

    receipt_id: Optional[str] = None

    prediction: Dict[str, Any] = Field(
        default_factory=dict
    )

    processing_time_ms: Optional[float] = None

    timestamp_utc: str

    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):

    success: bool

    total_requested: int

    total_successful: int

    total_failed: int

    predictions: List[Dict[str, Any]]

    timestamp_utc: str


class UploadPredictionResponse(BaseModel):

    success: bool

    filename: str

    receipt_id: Optional[str] = None

    prediction: Dict[str, Any] = Field(
        default_factory=dict
    )

    processing_time_ms: Optional[float] = None

    timestamp_utc: str

    error: Optional[str] = None


class ServiceInfoResponse(BaseModel):

    service: str
    version: str
    description: str
    pipeline: str
    stages: List[str]
    supported_extensions: List[str]
    max_upload_size_mb: int
    endpoints: List[str]


# ============================================================
# GLOBAL STATE HELPERS
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def ensure_directory(path: Path) -> Path:

    path.mkdir(
        parents=True,
        exist_ok=True
    )

    return path


def safe_float(
    value: Any,
    default: float = 0.0
) -> float:

    try:

        if value is None:
            return default

        result = float(value)

        if not result == result:
            return default

        if result in (
            float("inf"),
            float("-inf")
        ):
            return default

        return result

    except (
        TypeError,
        ValueError
    ):

        return default


def safe_int(
    value: Any,
    default: int = 0
) -> int:

    try:

        if value is None:
            return default

        return int(value)

    except (
        TypeError,
        ValueError
    ):

        return default


# ============================================================
# INFERENCE INITIALIZATION
# ============================================================

def initialize_inference() -> ReceiptInferencePipeline:

    configuration_manager = ConfigurationManager()

    inference_config = (
        configuration_manager
        .get_inference_config()
    )

    inference = ReceiptInferencePipeline(
        config=configuration_manager, inference_config=inference_config
    )

    return inference


# ============================================================
# RECEIPT PIPELINE ADAPTER
# ============================================================

def run_single_prediction(
    inference: ReceiptInferencePipeline,
    image_path: Path,
) -> Dict[str, Any]:

    image_path = image_path.resolve()

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not image_path.is_file():

        raise ValueError(
            f"Path is not a file: {image_path}"
        )

    if image_path.suffix.lower() not in ALLOWED_EXTENSIONS:

        raise ValueError(
            f"Unsupported image extension: "
            f"{image_path.suffix}"
        )

    if not hasattr(
        inference,
        "predict_one"
    ):

        raise RuntimeError(
            "ReceiptInferencePipeline "
            "does not expose predict_one()."
        )

    result = inference.predict_one(
        image_path
    )

    if result is None:

        raise RuntimeError(
            "Inference pipeline returned None."
        )

    if isinstance(
        result,
        dict
    ):

        return result

    if hasattr(
        result,
        "to_dict"
    ):

        converted = result.to_dict()

        if isinstance(
            converted,
            dict
        ):

            return converted

    if hasattr(
        result,
        "__dict__"
    ):

        return dict(
            result.__dict__
        )

    raise TypeError(
        "Unsupported inference result type: "
        f"{type(result).__name__}"
    )


def extract_receipt_id(
    prediction: Dict[str, Any],
    fallback: Optional[str] = None,
) -> Optional[str]:

    candidates = [
        prediction.get("receipt_id"),
        prediction.get("id"),
        prediction.get("filename"),
        fallback,
    ]

    for value in candidates:

        if value is not None:

            value = str(value).strip()

            if value:

                return value

    return None


# ============================================================
# UPLOAD VALIDATION
# ============================================================

def validate_upload(
    file: UploadFile,
) -> str:

    filename = (
        file.filename or ""
    ).strip()

    if not filename:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename."
        )

    extension = Path(
        filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: {extension}. "
                f"Supported extensions: "
                f"{sorted(ALLOWED_EXTENSIONS)}"
            )
        )

    if (
        file.content_type
        and file.content_type not in ALLOWED_CONTENT_TYPES
    ):

        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported content type: "
                f"{file.content_type}"
            )
        )

    return extension


async def save_upload_to_temp(
    file: UploadFile,
    temp_dir: Path,
) -> Path:

    extension = validate_upload(
        file
    )

    safe_name = (
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    target_path = (
        temp_dir / safe_name
    )

    total_size = 0

    with target_path.open(
        "wb"
    ) as output:

        while True:

            chunk = await file.read(
                1024 * 1024
            )

            if not chunk:
                break

            total_size += len(
                chunk
            )

            if (
                total_size
                > MAX_UPLOAD_SIZE_BYTES
            ):

                target_path.unlink(
                    missing_ok=True
                )

                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Uploaded file exceeds "
                        f"{MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
                    )
                )

            output.write(
                chunk
            )

    return target_path


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.inference = None
    app.state.model_loaded = False
    app.state.pipeline_ready = False
    app.state.model_path = None
    app.state.startup_error = None
    app.state.inference_lock = asyncio.Lock()

    startup_started = time.perf_counter()

    try:

        logger.info(
            "Starting Receipt OCR inference service."
        )

        logger.info(
            "Initializing receipt inference pipeline."
        )

        inference = initialize_inference()

        app.state.inference = inference

        app.state.model_loaded = True

        app.state.pipeline_ready = True

        pipeline_config = getattr(
            inference,
            "config",
            None
        )

        model_path = getattr(
            pipeline_config,
            "model_path",
            None
        )

        if model_path is not None:

            app.state.model_path = str(
                model_path
            )

        startup_time_ms = (
            time.perf_counter()
            - startup_started
        ) * 1000.0

        logger.info(
            "Receipt OCR inference service ready "
            "in %.2f ms.",
            startup_time_ms
        )

    except Exception as exc:

        app.state.inference = None

        app.state.model_loaded = False

        app.state.pipeline_ready = False

        app.state.model_path = None

        app.state.startup_error = str(
            exc
        )

        logger.exception(
            "Failed to initialize receipt inference pipeline."
        )

    yield

    logger.info(
        "Shutting down Receipt OCR inference service."
    )

    app.state.inference = None
    app.state.model_loaded = False
    app.state.pipeline_ready = False


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title=(
        "Receipt OCR & "
        "Document Intelligent Extraction API"
    ),

    description=(
        "Production-grade inference API for "
        "receipt image OCR, text cleaning, "
        "field extraction, confidence scoring, "
        "reliability analysis and JSON generation."
    ),

    version="1.0.0",

    lifespan=lifespan,

    docs_url="/api/docs",

    redoc_url="/api/redoc",

    openapi_url="/api/openapi.json"
)


# ============================================================
# PROMETHEUS
# ============================================================

Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
).instrument(
    app
).expose(
    app,
    endpoint="/metrics"
)


# ============================================================
# CORS
# ============================================================

cors_origins = os.getenv(
    "CORS_ORIGINS",
    ""
)

if cors_origins:

    allowed_origins = [
        origin.strip()
        for origin in cors_origins.split(",")
        if origin.strip()
    ]

else:

    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]


app.add_middleware(

    CORSMiddleware,

    allow_origins=allowed_origins,

    allow_credentials=True,

    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],

    allow_headers=[
        "Content-Type",
        "Authorization",
    ],
)


# ============================================================
# STATIC FILES
# ============================================================

if STATIC_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(
            directory=str(STATIC_DIR)
        ),
        name="static",
    )


# ============================================================
# HOME
# ============================================================

@app.get("/", response_class=HTMLResponse,)
async def home():

    if INDEX_HTML.exists():

        return HTMLResponse(
            INDEX_HTML.read_text(
                encoding="utf-8"
            )
        )

    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Receipt OCR API</title>
            <meta charset="utf-8">
        </head>
        <body>
            <h1>Receipt OCR & Document Intelligent Extraction API</h1>
            <p>Service is running.</p>
            <p>
                <a href="/api/docs">
                    Swagger Documentation
                </a>
            </p>
            <p>
                <a href="/health">
                    Health
                </a>
            </p>
        </body>
        </html>
        """
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health",response_model=InferenceHealthResponse,)
async def health():

    model_loaded = getattr(
        app.state,
        "model_loaded",
        False,
    )

    pipeline_ready = getattr(
        app.state,
        "pipeline_ready",
        False,
    )

    if pipeline_ready:

        status = "healthy"

    elif model_loaded:

        status = "degraded"

    else:

        status = "unhealthy"

    return InferenceHealthResponse(

        status=status,

        model_loaded=model_loaded,

        version=app.version,

        pipeline_ready=pipeline_ready,

        timestamp_utc=utc_now(),
    )


@app.get("/health/live")
async def liveness():

    return {
        "status": "alive",
        "timestamp_utc": utc_now(),
    }


@app.get("/health/ready")
async def readiness():

    if not getattr(
        app.state,
        "pipeline_ready",
        False,
    ):

        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "reason": getattr(
                    app.state,
                    "startup_error",
                    None,
                ),
            },
        )

    return {
        "status": "ready",
        "timestamp_utc": utc_now(),
    }


# ============================================================
# SERVICE INFO
# ============================================================

@app.get("/api/info", response_model=ServiceInfoResponse,)
async def service_info():

    return ServiceInfoResponse(

        service=(
            "Receipt OCR & "
            "Document Intelligent Extraction"
        ),

        version=app.version,

        description=(
            "End-to-end receipt inference service "
            "covering preprocessing, OCR, text cleaning, "
            "field extraction, confidence/reliability "
            "scoring and structured JSON generation."
        ),

        pipeline=(
            "ReceiptInferencePipeline"
        ),

        stages=[
            "Image Preprocessing",
            "OCR",
            "OCR Text Cleaning",
            "Field Extraction",
            "Confidence & Reliability",
            "JSON Generation",
        ],

        supported_extensions=sorted(
            ALLOWED_EXTENSIONS
        ),

        max_upload_size_mb=(
            MAX_UPLOAD_SIZE_BYTES
            // (1024 * 1024)
        ),

        endpoints=[
            "/predict",
            "/predict/upload",
            "/predict/batch",
            "/health",
            "/health/live",
            "/health/ready",
            "/metrics",
            "/api/docs",
        ],
    )


# ============================================================
# MODEL / PIPELINE INFO
# ============================================================

@app.get("/api/pipeline/info")
async def pipeline_info():

    inference = getattr(
        app.state,
        "inference",
        None,
    )

    if inference is None:

        raise HTTPException(
            status_code=503,
            detail="Inference pipeline unavailable."
        )

    config = getattr(
        inference,
        "config",
        None,
    )

    return {

        "success": True,

        "pipeline":
            "ReceiptInferencePipeline",

        "model_path":
            getattr(
                config,
                "model_path",
                None,
            ),

        "input_dir":
            str(
                getattr(
                    config,
                    "input_dir",
                    ""
                )
            ),

        "output_dir":
            str(
                getattr(
                    config,
                    "output_dir",
                    ""
                )
            ),

        "supported_extensions":
            sorted(
                ALLOWED_EXTENSIONS
            ),

        "timestamp_utc":
            utc_now(),
    }


# ============================================================
# PREDICT ONE IMAGE
# ============================================================

@app.post("/predict", response_model=PredictionResponse,)
async def predict(request: Request,image_path: str):

    request_id = str(
        uuid.uuid4()
    )

    if not getattr(
        app.state,
        "pipeline_ready",
        False,
    ):

        raise HTTPException(
            status_code=503,
            detail="Inference pipeline is not ready.",
            headers={
                "X-Request-ID": request_id
            },
        )

    source_path = Path(
        image_path
    ).expanduser().resolve()

    if not source_path.exists():

        raise HTTPException(
            status_code=404,
            detail=f"Image not found: {source_path}",
            headers={
                "X-Request-ID": request_id
            },
        )

    if not source_path.is_file():

        raise HTTPException(
            status_code=400,
            detail="Provided image path is not a file.",
            headers={
                "X-Request-ID": request_id
            },
        )

    if (source_path.suffix.lower() not in ALLOWED_EXTENSIONS):

        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported image extension: "
                f"{source_path.suffix}"
            ),
            headers={
                "X-Request-ID": request_id
            },
        )

    started = time.perf_counter()

    try:

        async with app.state.inference_lock:

            prediction = run_single_prediction(
                app.state.inference,
                source_path,
            )

        processing_time_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        receipt_id = extract_receipt_id(
            prediction,
            fallback=source_path.stem,
        )

        return PredictionResponse(

            success=True,

            receipt_id=receipt_id,

            prediction=prediction,

            processing_time_ms=round(
                processing_time_ms,
                2,
            ),

            timestamp_utc=utc_now(),

            error=None,
        )

    except FileNotFoundError as exc:

        logger.exception(
            "Prediction input not found. "
            "request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=404,
            detail=str(exc),
            headers={
                "X-Request-ID": request_id
            },
        )

    except ValueError as exc:

        logger.exception(
            "Invalid prediction input. "
            "request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
            headers={
                "X-Request-ID": request_id
            },
        )

    except Exception as exc:

        logger.exception(
            "Prediction failed. "
            "request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Receipt prediction failed."
            ),
            headers={
                "X-Request-ID": request_id
            },
        )


# ============================================================
# IMAGE UPLOAD
# ============================================================

@app.post("/predict/upload", response_model=UploadPredictionResponse,)
async def predict_upload(request: Request, file: UploadFile = File(...),):

    request_id = str(
        uuid.uuid4()
    )

    if not getattr(
        app.state,
        "pipeline_ready",
        False,
    ):

        raise HTTPException(
            status_code=503,
            detail="Inference pipeline is not ready.",
            headers={
                "X-Request-ID": request_id
            },
        )

    validate_upload(
        file
    )

    started = time.perf_counter()

    temporary_directory = TemporaryDirectory(
        prefix="receipt_api_"
    )

    try:

        temp_dir = Path(
            temporary_directory.name
        )

        staged_path = await save_upload_to_temp(
            file,
            temp_dir,
        )

        logger.info(
            "Processing uploaded receipt. "
            "request_id=%s filename=%s",
            request_id,
            file.filename,
        )

        async with app.state.inference_lock:

            prediction = run_single_prediction(
                app.state.inference,
                staged_path,
            )

        processing_time_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        receipt_id = extract_receipt_id(
            prediction,
            fallback=Path(
                file.filename
            ).stem,
        )

        return UploadPredictionResponse(

            success=True,

            filename=file.filename or staged_path.name,

            receipt_id=receipt_id,

            prediction=prediction,

            processing_time_ms=round(
                processing_time_ms,
                2,
            ),

            timestamp_utc=utc_now(),

            error=None,
        )

    except HTTPException:

        raise

    except ValueError as exc:

        logger.exception(
            "Invalid uploaded receipt. "
            "request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
            headers={
                "X-Request-ID": request_id
            },
        )

    except Exception:

        logger.exception(
            "Uploaded receipt inference failed. "
            "request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Uploaded receipt processing failed."
            ),
            headers={
                "X-Request-ID": request_id
            },
        )

    finally:

        await file.close()

        temporary_directory.cleanup()


# ============================================================
# BATCH PREDICTION
# ============================================================

@app.post("/predict/batch", response_model=BatchPredictionResponse,)
async def predict_batch(request: BatchInferenceRequest,):

    request_id = str(
        uuid.uuid4()
    )

    if not getattr(
        app.state,
        "pipeline_ready",
        False,
    ):

        raise HTTPException(
            status_code=503,
            detail="Inference pipeline is not ready.",
            headers={
                "X-Request-ID": request_id
            },
        )

    started = time.perf_counter()

    predictions = []

    successful = 0

    failed = 0

    async with app.state.inference_lock:

        for image in request.images:

            try:

                image_path = (
                    Path(
                        image
                    )
                    .expanduser()
                    .resolve()
                )

                prediction = run_single_prediction(
                    app.state.inference,
                    image_path,
                )

                receipt_id = extract_receipt_id(
                    prediction,
                    fallback=image_path.stem,
                )

                predictions.append({

                    "success": True,

                    "receipt_id": receipt_id,

                    "input_path": str(
                        image_path
                    ),

                    "prediction": prediction,

                    "error": None,
                })

                successful += 1

            except Exception as exc:

                logger.exception(
                    "Batch item failed. "
                    "request_id=%s image=%s",
                    request_id,
                    image,
                )

                predictions.append({

                    "success": False,

                    "receipt_id": Path(
                        image
                    ).stem,

                    "input_path": image,

                    "prediction": {},

                    "error": str(exc),
                })

                failed += 1

    logger.info(
        "Batch inference completed. "
        "request_id=%s total=%s successful=%s failed=%s "
        "time_ms=%.2f",
        request_id,
        len(request.images),
        successful,
        failed,
        (
            time.perf_counter()
            - started
        ) * 1000.0,
    )

    return BatchPredictionResponse(

        success=failed == 0,

        total_requested=len(
            request.images
        ),

        total_successful=successful,

        total_failed=failed,

        predictions=predictions,

        timestamp_utc=utc_now(),
    )


# ============================================================
# PREDICT LOCAL DIRECTORY
# ============================================================

@app.post("/predict/directory")
async def predict_directory(directory: str,):

    request_id = str(
        uuid.uuid4()
    )

    if not getattr(
        app.state,
        "pipeline_ready",
        False,
    ):

        raise HTTPException(
            status_code=503,
            detail="Inference pipeline is not ready.",
            headers={
                "X-Request-ID": request_id
            },
        )

    directory_path = (
        Path(
            directory
        )
        .expanduser()
        .resolve()
    )

    if not directory_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Directory not found.",
            headers={
                "X-Request-ID": request_id
            },
        )

    if not directory_path.is_dir():

        raise HTTPException(
            status_code=400,
            detail="Provided path is not a directory.",
            headers={
                "X-Request-ID": request_id
            },
        )

    image_paths = sorted(
        path
        for path in directory_path.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in ALLOWED_EXTENSIONS
        )
    )

    if not image_paths:

        raise HTTPException(
            status_code=400,
            detail="No supported receipt images found.",
            headers={
                "X-Request-ID": request_id
            },
        )

    started = time.perf_counter()

    predictions = []

    successful = 0

    failed = 0

    async with app.state.inference_lock:

        for image_path in image_paths:

            try:

                prediction = run_single_prediction(
                    app.state.inference,
                    image_path,
                )

                predictions.append({

                    "success": True,

                    "receipt_id":
                        extract_receipt_id(
                            prediction,
                            fallback=image_path.stem,
                        ),

                    "input_path":
                        str(image_path),

                    "prediction":
                        prediction,

                    "error":
                        None,
                })

                successful += 1

            except Exception as exc:

                logger.exception(
                    "Directory inference failed. "
                    "request_id=%s image=%s",
                    request_id,
                    image_path,
                )

                predictions.append({

                    "success": False,

                    "receipt_id":
                        image_path.stem,

                    "input_path":
                        str(image_path),

                    "prediction":
                        {},

                    "error":
                        str(exc),
                })

                failed += 1

    return {

        "success":
            failed == 0,

        "directory":
            str(directory_path),

        "total_requested":
            len(image_paths),

        "total_successful":
            successful,

        "total_failed":
            failed,

        "predictions":
            predictions,

        "processing_time_ms":
            round(
                (
                    time.perf_counter()
                    - started
                ) * 1000.0,
                2,
            ),

        "timestamp_utc":
            utc_now(),
    }


# ============================================================
# CONFIG / ARTIFACT STATUS
# ============================================================

@app.get("/api/artifacts")
async def artifact_status():

    inference = getattr(
        app.state,
        "inference",
        None,
    )

    if inference is None:

        return {

            "success": False,

            "pipeline_ready": False,

            "artifacts": {},

            "timestamp_utc":
                utc_now(),
        }

    config = getattr(
        inference,
        "config",
        None,
    )

    fields = [
        "root_dir",
        "input_dir",
        "output_dir",
        "model_path",
    ]

    artifacts = {}

    for field in fields:

        value = getattr(
            config,
            field,
            None,
        )

        if value is None:

            continue

        path = Path(
            value
        )

        artifacts[field] = {

            "path":
                str(path),

            "exists":
                path.exists(),

            "is_file":
                path.is_file()
                if path.exists()
                else False,

            "is_directory":
                path.is_dir()
                if path.exists()
                else False,
        }

    return {

        "success": True,

        "pipeline_ready":
            getattr(
                app.state,
                "pipeline_ready",
                False,
            ),

        "artifacts":
            artifacts,

        "timestamp_utc":
            utc_now(),
    }


# ============================================================
# VALIDATION ERROR HANDLER
# ============================================================

@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request,exc: RequestValidationError,):

    return JSONResponse(

        status_code=422,

        content={

            "success": False,

            "detail":
                "Request validation failed.",

            "errors":
                exc.errors(),

            "timestamp_utc":
                utc_now(),
        },
    )


# ============================================================
# HTTP EXCEPTION HANDLER
# ============================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request,exc: HTTPException,):

    headers = dict(
        exc.headers or {}
    )

    request_id = headers.pop(
        "X-Request-ID",
        None,
    )

    content = {

        "success": False,

        "detail":
            exc.detail,

        "status_code":
            exc.status_code,

        "timestamp_utc":
            utc_now(),
    }

    if request_id:

        content["request_id"] = request_id

    return JSONResponse(

        status_code=exc.status_code,

        content=content,

        headers=exc.headers,
    )


# ============================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception,):

    request_id = str(
        uuid.uuid4()
    )

    logger.exception(
        "Unhandled application exception. "
        "request_id=%s path=%s",
        request_id,
        request.url.path,
    )

    return JSONResponse(

        status_code=500,

        content={

            "success": False,

            "detail":
                "Internal Server Error",

            "request_id":
                request_id,

            "timestamp_utc":
                utc_now(),
        },
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    uvicorn.run(

        "app:app",

        host=os.getenv(
            "HOST",
            "0.0.0.0",
        ),

        port=int(
            os.getenv(
                "PORT",
                "8000",
            )
        ),

        reload=False,

        access_log=True,

        log_level=os.getenv(
            "LOG_LEVEL",
            "info",
        ),
    )