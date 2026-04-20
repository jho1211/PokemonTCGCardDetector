from __future__ import annotations

import argparse
import json
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_BASE_URL = "https://api.tcgdex.net/v2"
DEFAULT_LANGUAGE = "en"
DEFAULT_SERIES_PREFIXES = ("sv", "swsh")


def _fetch_json(url: str, timeout: float = 20.0) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "PokemonTCGCardDetector/1.0"})
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_bytes(url: str, timeout: float = 20.0) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "PokemonTCGCardDetector/1.0"})
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        body = response.read()
        content_type = response.headers.get("Content-Type", "")
        return body, content_type


def _is_supported_set(set_payload: dict[str, Any], prefixes: tuple[str, ...]) -> bool:
    set_id = str(set_payload.get("id", "")).lower()
    if any(set_id.startswith(prefix.lower()) for prefix in prefixes):
        return True

    serie = set_payload.get("serie", {})
    serie_id = str(serie.get("id", "")).lower()
    if any(serie_id.startswith(prefix.lower()) for prefix in prefixes):
        return True

    return False


def _fetch_symbol_png(symbol_url: str) -> bytes | None:
    candidates = [
        symbol_url,
        f"{symbol_url}.png" if not symbol_url.endswith(".png") else symbol_url,
    ]

    for url in candidates:
        try:
            content, _ = _fetch_bytes(url)
        except urllib.error.URLError:
            continue

        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            continue

        success, encoded = cv2.imencode(".png", image)
        if not success:
            continue

        return encoded.tobytes()

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Download TCGdex set symbols for template matching.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="TCGdex API base URL (without /en suffix).")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="Language code for set metadata.")
    parser.add_argument(
        "--series-prefixes",
        default=",".join(DEFAULT_SERIES_PREFIXES),
        help="Comma-separated set/series prefixes to include (e.g. sv,swsh).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "templates" / "set_symbols"),
        help="Directory to store set symbol PNG templates and metadata.",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    if base_url.endswith(f"/{args.language}"):
        base_url = base_url[: -(len(args.language) + 1)]

    prefixes = tuple(filter(None, [value.strip() for value in args.series_prefixes.split(",")]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sets_url = f"{base_url}/{args.language}/sets"
    print(f"Fetching sets from: {sets_url}")

    try:
        sets_payload = _fetch_json(sets_url)
    except Exception as exc:
        print(f"Failed to fetch set list: {exc}")
        return 2

    if not isinstance(sets_payload, list):
        print("Unexpected sets response format.")
        return 2

    selected_sets = [item for item in sets_payload if isinstance(item, dict) and _is_supported_set(item, prefixes)]
    print(f"Selected {len(selected_sets)} sets using prefixes: {', '.join(prefixes)}")

    metadata: list[dict[str, str]] = []
    downloaded = 0

    for entry in selected_sets:
        set_id = str(entry.get("id", "")).strip()
        set_name = str(entry.get("name", set_id)).strip()

        if not set_id:
            continue

        symbol_url = str(entry.get("symbol", "")).strip()
        if not symbol_url:
            try:
                details = _fetch_json(f"{base_url}/{args.language}/sets/{set_id}")
                if isinstance(details, dict):
                    symbol_url = str(details.get("symbol", "")).strip()
            except Exception:
                symbol_url = ""

        if not symbol_url:
            print(f"Skipping {set_id}: missing symbol URL")
            continue

        image_data = _fetch_symbol_png(symbol_url)
        if image_data is None:
            print(f"Skipping {set_id}: unable to fetch usable symbol image")
            continue

        file_name = f"{set_id}.png"
        file_path = output_dir / file_name
        file_path.write_bytes(image_data)

        metadata.append(
            {
                "set_id": set_id,
                "set_name": set_name,
                "template_file": file_name,
                "symbol_url": symbol_url,
            }
        )
        downloaded += 1
        print(f"Downloaded symbol for {set_id} -> {file_name}")

    metadata.sort(key=lambda item: item["set_id"])
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Downloaded {downloaded} symbols")
    print(f"Metadata written to: {metadata_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
