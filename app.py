"""
ColorArt Backend — FastAPI Application
Image processing endpoint for color-by-numbers segmentation.
"""

import asyncio
import io
from functools import partial

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from processing.pipeline import process_image

# ─── Constants ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE = 5 * 1024 * 1024          # 5 MB
MAX_RESOLUTION = 6000                     # 6000px in any dimension
PROCESSING_TIMEOUT_SECONDS = 10           # Abort if processing exceeds this

# ─── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ColorArt Image Processing API",
    version="1.0.0",
    description="Converts uploaded images into color-by-numbers vector region data.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/api/v1/process")
async def process_image_endpoint(image: UploadFile = File(...)):
    """
    Accept an uploaded image file, process it through the OpenCV pipeline,
    and return structured JSON with vector region paths, palette, and metadata.
    """

    # ── Validate file type ─────────────────────────────────────────────────────
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # ── Read and validate size ─────────────────────────────────────────────────
    contents = await image.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB",
        )

    # ── Validate resolution ────────────────────────────────────────────────────
    try:
        pil_image = Image.open(io.BytesIO(contents))
        original_width, original_height = pil_image.size
        if original_width > MAX_RESOLUTION or original_height > MAX_RESOLUTION:
            raise HTTPException(
                status_code=400,
                detail=f"Image resolution too high. Maximum is {MAX_RESOLUTION}px per side.",
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # ── Process with timeout ───────────────────────────────────────────────────
    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                partial(process_image, contents, original_width, original_height),
            ),
            timeout=PROCESSING_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail="Image processing timed out. Try a simpler image.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
