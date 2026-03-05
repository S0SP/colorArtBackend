# ColorArt Backend

Image processing backend for the ColorArt color-by-numbers app.

Converts uploaded photos into structured vector region data using OpenCV.

## Setup

```bash
cd colorArtBackend
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Server starts on `http://0.0.0.0:5000`.

## API

### `POST /api/v1/process`

Upload an image as multipart form-data (`image` field).

**Limits:**
- Max file size: 5MB
- Max resolution: 6000px per side
- Processing timeout: 10 seconds

**Response:** JSON with `meta`, `width`, `height`, `outlinePath`, `regions[]`, and `palette[]`.

### `GET /health`

Returns `{"status": "ok"}`.
