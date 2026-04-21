# Pokemon TCG Identifier

This repository now includes the first implementation slice for both backend and mobile:

- FastAPI backend with `POST /identify-card`
- Expo React Native app with camera capture, upload, result rendering, and local scan history

## Backend (FastAPI)

### Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

.env file
```
YOLO_MODEL_PATH="data/yolo26n-obb-pokemon-card.pt"
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
```


### Run

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then start the API:

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

- `GET /health`
- `POST /identify-card` with multipart field `image`

Current behavior for `POST /identify-card` returns mapped data from [example_tcgapi_response.json](example_tcgapi_response.json) so the mobile app can be integrated before CV/OCR services are fully implemented.

## Mobile (Expo React Native)

### Setup

```bash
cd mobile
npm install
```

### Configure API URL

Default API URL is in [mobile/app.json](mobile/app.json):

- `expo.extra.apiBaseUrl`

For iOS simulator, `http://localhost:8000` usually works.
For physical devices, use your machine LAN IP, for example `http://192.168.1.20:8000`.

### Run

```bash
npm run start
```

Then open in Expo Go or simulator.

## Implemented User Fields

The result screen currently displays:

- Card Name
- Collection
- Collector Number
- Market Price (USD, from `pricing.tcgplayer.normal.marketPrice`)
- Card image

## Next Implementation Targets

1. Replace mocked identify logic with CV + OCR pipeline.
2. Add confidence thresholds (`high`, `medium`, `low`) from real scoring.
3. Add richer error payloads for low-confidence and no-match cases.
4. Add gallery picker support (optional, post-MVP).
