from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tcgdexsdk import Query, TCGdex  # type: ignore[reportMissingImports]

try:
    from tcgdexsdk.enums import Extension, Quality  # type: ignore[reportMissingImports]
except Exception:
    Extension = None  # type: ignore[assignment]
    Quality = None  # type: ignore[assignment]

DEFAULT_API_BASE = "https://api.tcgdex.net/v2"
DEFAULT_LANGUAGE = "en"
DEFAULT_CLASS_NAME = "pokemon_card"
DEFAULT_BACKGROUND_TEMPLATE = "https://yavuzceliker.github.io/sample-images/image-{index}.jpg"
DEFAULT_BACKGROUND_MIN_INDEX = 1
DEFAULT_BACKGROUND_MAX_INDEX = 2000

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class SourceCard:
    set_id: str
    card_id: str
    image_url: str
    local_path: Path


@dataclass
class PlacedCard:
    source: SourceCard
    quad: np.ndarray
    visible_ratio: float
    bbox: tuple[float, float, float, float]


@dataclass
class SceneResult:
    image: np.ndarray
    labels: list[str]
    cards: list[PlacedCard]
    requested_count: int
    layout_mode: str
    background_index: int
    background_url: str


def _parse_args() -> argparse.Namespace:
    backend_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate synthetic YOLO OBB dataset for pokemon card detection.")
    parser.add_argument("--output-root", default=str(backend_root / "data" / "obb_synth"), help="Dataset output root directory.")
    parser.add_argument(
        "--metadata-path",
        default=str(backend_root / "templates" / "set_symbols" / "metadata.json"),
        help="Path to set symbol metadata.json used for source set IDs.",
    )
    parser.add_argument("--exclude-set-ids", default="swshp", help="Comma-separated set IDs to exclude.")
    parser.add_argument("--cards-per-set", type=int, default=25, help="Target number of source PNG cards to cache per set.")
    parser.add_argument("--samples", type=int, default=10_000, help="Number of synthetic samples to generate.")
    parser.add_argument("--img-size", type=int, default=640, help="Square output image size in pixels.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--layout-grid-ratio", type=float, default=0.2, help="Probability of generating a grid layout.")
    parser.add_argument("--max-bg-retries", type=int, default=12, help="Max background fetch attempts per sample.")
    parser.add_argument("--min-visible-ratio", type=float, default=0.6, help="Minimum visible card area ratio to keep a placement.")
    parser.add_argument("--max-overlap-iou", type=float, default=0.2, help="Maximum IoU overlap allowed between card placements.")
    parser.add_argument("--placement-attempts", type=int, default=40, help="Attempts per card placement before skipping that card.")
    parser.add_argument("--min-bg-size", type=int, default=320, help="Minimum background width/height required before resize.")
    parser.add_argument("--perspective-strength", type=float, default=0.15, help="Max perspective corner jitter relative to card min dimension.")
    parser.add_argument("--tcgdex-base-url", default=DEFAULT_API_BASE, help="TCGdex API base URL.")
    parser.add_argument("--tcgdex-language", default=DEFAULT_LANGUAGE, help="TCGdex language code.")
    parser.add_argument("--class-name", default=DEFAULT_CLASS_NAME, help="Single class name used in dataset.yaml.")
    parser.add_argument("--background-url-template", default=DEFAULT_BACKGROUND_TEMPLATE, help="Template URL for sampled backgrounds.")
    parser.add_argument("--background-min-index", type=int, default=DEFAULT_BACKGROUND_MIN_INDEX, help="Min random background index.")
    parser.add_argument("--background-max-index", type=int, default=DEFAULT_BACKGROUND_MAX_INDEX, help="Max random background index.")
    parser.add_argument("--request-timeout", type=float, default=20.0, help="HTTP timeout in seconds.")
    parser.add_argument("--card-page-size", type=int, default=250, help="TCGdex query page size for set card listing.")
    parser.add_argument("--max-set-pages", type=int, default=20, help="Max pages per set during card listing.")
    return parser.parse_args()


def _prepare_output_dirs(output_root: Path) -> dict[str, Path]:
    paths = {
        "images_train": output_root / "images" / "train",
        "images_val": output_root / "images" / "val",
        "images_test": output_root / "images" / "test",
        "labels_train": output_root / "labels" / "train",
        "labels_val": output_root / "labels" / "val",
        "labels_test": output_root / "labels" / "test",
        "cache_cards": output_root / "cache" / "cards",
        "cache_backgrounds": output_root / "cache" / "backgrounds",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _load_set_ids(metadata_path: Path, excluded_set_ids: set[str]) -> list[str]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Set metadata not found: {metadata_path}")

    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Set metadata must be a list of objects")

    set_ids: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        set_id = str(item.get("set_id", "")).strip()
        if not set_id:
            continue
        if set_id.lower() in excluded_set_ids:
            continue
        set_ids.append(set_id)

    deduped: list[str] = []
    seen: set[str] = set()
    for set_id in set_ids:
        key = set_id.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(set_id)

    return deduped


async def _list_cards_for_set(
    client: TCGdex,
    set_id: str,
    page_size: int,
    max_pages: int,
) -> list[Any]:
    cards: list[Any] = []
    page = 1

    while page <= max_pages:
        query = Query().equal("set.id", set_id).paginate(page=page, itemsPerPage=page_size)
        batch = await client.card.list(query)
        if not batch:
            break

        cards.extend(batch)
        if len(batch) < page_size:
            break

        page += 1

    return cards


def _safe_field(payload: Any, key: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _fetch_bytes(url: str, timeout: float) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "PokemonTCGCardDetector/1.0"})
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        body = response.read()
        content_type = response.headers.get("Content-Type", "")
        return body, content_type


def _is_png_bytes(content: bytes) -> bool:
    return len(content) >= len(PNG_SIGNATURE) and content.startswith(PNG_SIGNATURE)


def _call_card_image_method(card: Any, method_name: str) -> Any | None:
    method = getattr(card, method_name, None)
    if not callable(method):
        return None

    quality_value = getattr(Quality, "HIGH", "high") if Quality is not None else "high"
    extension_value = getattr(Extension, "PNG", "png") if Extension is not None else "png"

    candidates = [
        lambda: method(quality_value, extension_value),
        lambda: method(quality=quality_value, extension=extension_value),
        lambda: method("high", "png"),
        lambda: method(quality="high", extension="png"),
        lambda: method(),
    ]

    for candidate in candidates:
        try:
            value = candidate()
        except TypeError:
            continue
        except Exception:
            continue

        if value is not None:
            return value

    return None


def _coerce_bytes(value: Any) -> bytes | None:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    return None


def _normalize_png_bytes(content: bytes) -> tuple[bytes | None, str | None]:
    if _is_png_bytes(content):
        return content, None

    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        return None, "decode_failed"

    success, encoded = cv2.imencode(".png", image)
    if not success:
        return None, "encode_failed"

    return encoded.tobytes(), None


async def _cache_card_png(
    client: TCGdex,
    card_id: str,
    card_payload: Any,
    output_path: Path,
    timeout: float,
) -> tuple[bool, str | None, str | None]:
    card = card_payload
    if not hasattr(card, "get_image"):
        try:
            card = await client.card.get(card_id)
        except Exception as exc:
            return False, None, f"card_get_error:{type(exc).__name__}"

    if card is None:
        return False, None, "card_not_found"

    image_url_value = _call_card_image_method(card, "get_image_url")
    resolved_url = str(image_url_value).strip() if image_url_value is not None else ""

    image_value = _call_card_image_method(card, "get_image")
    image_bytes = _coerce_bytes(image_value)

    if image_bytes is None:
        if not resolved_url:
            raw_image_url = str(_safe_field(card, "image", "")).strip()
            if raw_image_url:
                resolved_url = raw_image_url if raw_image_url.lower().endswith(".png") else f"{raw_image_url}.png"

        if not resolved_url:
            return False, None, "image_unavailable"

        try:
            image_bytes, _ = _fetch_bytes(resolved_url, timeout=timeout)
        except Exception as exc:
            return False, resolved_url or None, f"download_error:{type(exc).__name__}"

    normalized_png, normalize_error = _normalize_png_bytes(image_bytes)
    if normalized_png is None:
        return False, resolved_url or None, normalize_error or "png_normalize_failed"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(normalized_png)
    return True, resolved_url or None, None


async def _build_source_card_pool(
    set_ids: list[str],
    cards_cache_dir: Path,
    cards_per_set: int,
    seed: int,
    base_url: str,
    language: str,
    timeout: float,
    page_size: int,
    max_pages: int,
) -> tuple[list[SourceCard], dict[str, Any]]:
    rng = random.Random(seed)

    client = TCGdex()
    client.setEndpoint(base_url.rstrip("/"))

    tasks = [
        _list_cards_for_set(client=client, set_id=set_id, page_size=page_size, max_pages=max_pages)
        for set_id in set_ids
    ]
    fetched_per_set = await asyncio.gather(*tasks, return_exceptions=True)

    source_cards: list[SourceCard] = []
    summary: dict[str, Any] = {
        "sets": {},
        "warnings": [],
        "language": language,
        "base_url": base_url,
    }

    for set_id, payload in zip(set_ids, fetched_per_set, strict=True):
        set_info = {
            "listed_cards": 0,
            "downloaded_cards": 0,
            "skipped": {},
        }

        if isinstance(payload, Exception):
            summary["warnings"].append(f"set {set_id}: list_failed {type(payload).__name__}: {payload}")
            summary["sets"][set_id] = set_info
            continue

        cards = list(payload)
        set_info["listed_cards"] = len(cards)
        rng.shuffle(cards)

        downloaded_for_set = 0
        for card in cards:
            card_id = str(_safe_field(card, "id", "")).strip()
            image_url = str(_safe_field(card, "image", "")).strip()
            if not card_id:
                key = "missing_card_id"
                set_info["skipped"][key] = int(set_info["skipped"].get(key, 0)) + 1
                continue

            output_path = cards_cache_dir / set_id / f"{card_id}.png"
            if output_path.exists():
                source_cards.append(SourceCard(set_id=set_id, card_id=card_id, image_url=image_url, local_path=output_path))
                downloaded_for_set += 1
            else:
                ok, resolved_url, reason = await _cache_card_png(
                    client=client,
                    card_id=card_id,
                    card_payload=card,
                    output_path=output_path,
                    timeout=timeout,
                )
                if not ok:
                    key = reason or "download_failed"
                    set_info["skipped"][key] = int(set_info["skipped"].get(key, 0)) + 1
                    continue

                source_cards.append(
                    SourceCard(
                        set_id=set_id,
                        card_id=card_id,
                        image_url=resolved_url or image_url,
                        local_path=output_path,
                    )
                )
                downloaded_for_set += 1

            if downloaded_for_set >= cards_per_set:
                break

        set_info["downloaded_cards"] = downloaded_for_set
        if downloaded_for_set < cards_per_set:
            summary["warnings"].append(
                f"set {set_id}: only {downloaded_for_set} PNG cards available (target {cards_per_set})"
            )

        summary["sets"][set_id] = set_info

    summary["total_source_cards"] = len(source_cards)
    return source_cards, summary


@lru_cache(maxsize=256)
def _load_card_image(path_text: str) -> np.ndarray | None:
    image = cv2.imread(path_text, cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        return None
    return image


def _sample_card_count(rng: random.Random) -> int:
    roll = rng.random()
    if roll < 0.85:
        return rng.randint(1, 8)
    if roll < 0.97:
        return rng.randint(9, 16)
    return rng.randint(17, 30)


def _sample_target_height(rng: random.Random, img_size: int, requested_count: int) -> float:
    if requested_count <= 2:
        return rng.uniform(0.36, 0.58) * img_size
    if requested_count <= 8:
        return rng.uniform(0.20, 0.42) * img_size
    if requested_count <= 16:
        return rng.uniform(0.14, 0.28) * img_size
    return rng.uniform(0.09, 0.20) * img_size


def _augment_card_image(image: np.ndarray, np_rng: np.random.Generator) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 4:
        bgr = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3].copy()
    elif image.ndim == 3 and image.shape[2] == 3:
        bgr = image.astype(np.float32)
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
    else:
        converted = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        bgr = converted.astype(np.float32)
        alpha = np.full(converted.shape[:2], 255, dtype=np.uint8)

    contrast = float(np_rng.uniform(0.90, 1.10))
    brightness = float(np_rng.uniform(-12.0, 12.0))
    bgr = np.clip((bgr * contrast) + brightness, 0, 255)

    hsv = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + float(np_rng.uniform(-4.0, 4.0))) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(np_rng.uniform(0.90, 1.10)), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * float(np_rng.uniform(0.92, 1.08)), 0, 255)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    sigma = float(np_rng.uniform(0.0, 4.0))
    if sigma > 0.01:
        noise = np_rng.normal(0.0, sigma, size=bgr.shape).astype(np.float32)
        bgr = np.clip(bgr + noise, 0, 255)

    out = np.dstack([bgr.astype(np.uint8), alpha])
    return out


def _fit_background_to_canvas(background: np.ndarray, img_size: int) -> np.ndarray:
    h, w = background.shape[:2]
    scale = max(img_size / float(w), img_size / float(h))
    resized = cv2.resize(background, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)

    new_h, new_w = resized.shape[:2]
    start_x = max(0, (new_w - img_size) // 2)
    start_y = max(0, (new_h - img_size) // 2)
    cropped = resized[start_y : start_y + img_size, start_x : start_x + img_size]

    if cropped.shape[0] != img_size or cropped.shape[1] != img_size:
        cropped = cv2.resize(cropped, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    return cropped


def _get_background(
    rng: random.Random,
    backgrounds_cache_dir: Path,
    min_bg_size: int,
    max_retries: int,
    timeout: float,
    url_template: str,
    min_index: int,
    max_index: int,
    img_size: int,
) -> tuple[int, str, np.ndarray] | None:
    for _ in range(max_retries):
        bg_index = rng.randint(min_index, max_index)
        bg_url = url_template.format(index=bg_index)
        cache_path = backgrounds_cache_dir / f"image-{bg_index}.jpg"

        background: np.ndarray | None = None
        if cache_path.exists():
            background = cv2.imread(str(cache_path), cv2.IMREAD_COLOR)

        if background is None or background.size == 0:
            try:
                content, _ = _fetch_bytes(bg_url, timeout=timeout)
            except Exception:
                continue

            decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None or decoded.size == 0:
                continue

            if min(decoded.shape[:2]) < min_bg_size:
                continue

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cache_path), decoded)
            background = decoded

        if min(background.shape[:2]) < min_bg_size:
            continue

        prepared = _fit_background_to_canvas(background, img_size=img_size)
        return bg_index, bg_url, prepared

    return None


def _grid_centers(count: int, img_size: int, rng: random.Random) -> list[tuple[float, float]]:
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = max(1, math.ceil(count / cols))
    cell_w = img_size / float(cols)
    cell_h = img_size / float(rows)

    centers: list[tuple[float, float]] = []
    for idx in range(count):
        row = idx // cols
        col = idx % cols

        cx = (col + 0.5) * cell_w
        cy = (row + 0.5) * cell_h

        jitter_x = rng.uniform(-0.18 * cell_w, 0.18 * cell_w)
        jitter_y = rng.uniform(-0.18 * cell_h, 0.18 * cell_h)
        centers.append((cx + jitter_x, cy + jitter_y))

    return centers


def _propose_center(
    rng: random.Random,
    img_size: int,
    layout_mode: str,
    grid_centers: list[tuple[float, float]] | None,
    slot_index: int,
) -> tuple[float, float]:
    if layout_mode == "grid" and grid_centers is not None and slot_index < len(grid_centers):
        base_x, base_y = grid_centers[slot_index]
        return base_x + rng.uniform(-12.0, 12.0), base_y + rng.uniform(-12.0, 12.0)

    return (
        rng.uniform(-0.10 * img_size, 1.10 * img_size),
        rng.uniform(-0.10 * img_size, 1.10 * img_size),
    )


def _build_target_quad(
    center_x: float,
    center_y: float,
    target_w: float,
    target_h: float,
    perspective_strength: float,
    rng: random.Random,
) -> np.ndarray:
    base = np.array(
        [
            [-target_w / 2.0, -target_h / 2.0],
            [target_w / 2.0, -target_h / 2.0],
            [target_w / 2.0, target_h / 2.0],
            [-target_w / 2.0, target_h / 2.0],
        ],
        dtype=np.float32,
    )

    angle = rng.uniform(-85.0, 85.0)
    angle_rad = math.radians(angle)
    rot = np.array(
        [
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)],
        ],
        dtype=np.float32,
    )

    rotated = base @ rot.T

    jitter = perspective_strength * min(target_w, target_h)
    perspective_noise = np.array(
        [
            [rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)],
            [rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)],
            [rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)],
            [rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)],
        ],
        dtype=np.float32,
    )

    quad = rotated + perspective_noise
    quad[:, 0] += center_x
    quad[:, 1] += center_y
    return quad.astype(np.float32)


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]

    edge_a = ordered[1] - ordered[0]
    edge_b = ordered[2] - ordered[0]
    cross_z = (edge_a[0] * edge_b[1]) - (edge_a[1] * edge_b[0])
    if cross_z < 0:
        ordered[[1, 3]] = ordered[[3, 1]]

    return ordered


def _quad_bbox(quad: np.ndarray) -> tuple[float, float, float, float]:
    min_xy = np.min(quad, axis=0)
    max_xy = np.max(quad, axis=0)
    return float(min_xy[0]), float(min_xy[1]), float(max_xy[0]), float(max_xy[1])


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter_area
    if union <= 1e-8:
        return 0.0

    return inter_area / union


def _visible_ratio(quad: np.ndarray, img_size: int) -> float:
    ordered = _order_quad_points(quad)
    full_area = abs(float(cv2.contourArea(ordered)))
    if full_area <= 1.0:
        return 0.0

    canvas = np.array(
        [[0.0, 0.0], [img_size - 1.0, 0.0], [img_size - 1.0, img_size - 1.0], [0.0, img_size - 1.0]],
        dtype=np.float32,
    )

    inter_area, _ = cv2.intersectConvexConvex(ordered.astype(np.float32), canvas)
    if inter_area <= 0.0:
        return 0.0

    return float(inter_area / full_area)


def _quad_to_label_line(quad: np.ndarray, img_size: int, class_index: int = 0) -> str:
    clipped = np.clip(quad.copy(), 0.0, float(img_size - 1))
    ordered = _order_quad_points(clipped)

    area = abs(float(cv2.contourArea(ordered)))
    if area <= 2.0:
        raise ValueError("degenerate_quad")

    normalized = ordered.copy()
    normalized[:, 0] = normalized[:, 0] / float(img_size)
    normalized[:, 1] = normalized[:, 1] / float(img_size)

    flat = normalized.reshape(-1)
    values = " ".join(f"{float(v):.6f}" for v in flat.tolist())
    return f"{class_index} {values}"


def _blend_card_on_canvas(canvas: np.ndarray, card_rgba: np.ndarray, dst_quad: np.ndarray) -> np.ndarray:
    h, w = card_rgba.shape[:2]
    src_quad = np.array([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]], dtype=np.float32)

    if card_rgba.shape[2] == 4:
        card_bgr = card_rgba[:, :, :3]
        alpha = card_rgba[:, :, 3]
    else:
        card_bgr = card_rgba[:, :, :3]
        alpha = np.full((h, w), 255, dtype=np.uint8)

    matrix = cv2.getPerspectiveTransform(src_quad, dst_quad.astype(np.float32))

    warped_card = cv2.warpPerspective(
        card_bgr,
        matrix,
        (canvas.shape[1], canvas.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_alpha = cv2.warpPerspective(
        alpha,
        matrix,
        (canvas.shape[1], canvas.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    alpha_f = (warped_alpha.astype(np.float32) / 255.0)[..., None]
    out = (warped_card.astype(np.float32) * alpha_f) + (canvas.astype(np.float32) * (1.0 - alpha_f))
    return np.clip(out, 0, 255).astype(np.uint8)


def _compose_scene(
    rng: random.Random,
    np_rng: np.random.Generator,
    source_cards: list[SourceCard],
    backgrounds_cache_dir: Path,
    img_size: int,
    min_bg_size: int,
    max_bg_retries: int,
    timeout: float,
    bg_url_template: str,
    bg_min_index: int,
    bg_max_index: int,
    max_overlap_iou: float,
    min_visible_ratio: float,
    perspective_strength: float,
    placement_attempts: int,
    layout_grid_ratio: float,
) -> SceneResult | None:
    bg = _get_background(
        rng=rng,
        backgrounds_cache_dir=backgrounds_cache_dir,
        min_bg_size=min_bg_size,
        max_retries=max_bg_retries,
        timeout=timeout,
        url_template=bg_url_template,
        min_index=bg_min_index,
        max_index=bg_max_index,
        img_size=img_size,
    )
    if bg is None:
        return None

    bg_index, bg_url, canvas = bg
    requested_count = _sample_card_count(rng)
    layout_mode = "grid" if rng.random() < layout_grid_ratio else "free"
    grid_centers = _grid_centers(requested_count, img_size=img_size, rng=rng) if layout_mode == "grid" else None

    labels: list[str] = []
    placed_cards: list[PlacedCard] = []
    bboxes: list[tuple[float, float, float, float]] = []

    for slot_index in range(requested_count):
        placed = False
        for _ in range(placement_attempts):
            source = rng.choice(source_cards)
            card_image = _load_card_image(str(source.local_path))
            if card_image is None:
                continue

            if card_image.ndim == 2:
                card_image = cv2.cvtColor(card_image, cv2.COLOR_GRAY2BGRA)
            elif card_image.ndim == 3 and card_image.shape[2] == 3:
                alpha = np.full(card_image.shape[:2], 255, dtype=np.uint8)
                card_image = np.dstack([card_image, alpha])

            card_aug = _augment_card_image(card_image, np_rng=np_rng)

            source_h, source_w = card_aug.shape[:2]
            center_x, center_y = _propose_center(
                rng=rng,
                img_size=img_size,
                layout_mode=layout_mode,
                grid_centers=grid_centers,
                slot_index=slot_index,
            )

            target_h = _sample_target_height(rng, img_size=img_size, requested_count=requested_count)
            source_ratio = source_w / max(1.0, float(source_h))
            target_w = target_h * source_ratio

            quad = _build_target_quad(
                center_x=center_x,
                center_y=center_y,
                target_w=target_w,
                target_h=target_h,
                perspective_strength=perspective_strength,
                rng=rng,
            )

            visible = _visible_ratio(quad, img_size=img_size)
            if visible < min_visible_ratio:
                continue

            clipped_quad = np.clip(quad, 0.0, float(img_size - 1)).astype(np.float32)
            ordered = _order_quad_points(clipped_quad)
            if abs(float(cv2.contourArea(ordered))) <= 2.0:
                continue

            bbox = _quad_bbox(ordered)
            if any(_bbox_iou(bbox, other) > max_overlap_iou for other in bboxes):
                continue

            try:
                label_line = _quad_to_label_line(ordered, img_size=img_size, class_index=0)
            except ValueError:
                continue

            canvas = _blend_card_on_canvas(canvas=canvas, card_rgba=card_aug, dst_quad=quad)
            labels.append(label_line)
            bboxes.append(bbox)
            placed_cards.append(
                PlacedCard(
                    source=source,
                    quad=ordered,
                    visible_ratio=visible,
                    bbox=bbox,
                )
            )
            placed = True
            break

        if not placed and layout_mode == "grid":
            continue

    if not labels:
        return None

    if np_rng.random() < 0.2:
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    return SceneResult(
        image=canvas,
        labels=labels,
        cards=placed_cards,
        requested_count=requested_count,
        layout_mode=layout_mode,
        background_index=bg_index,
        background_url=bg_url,
    )


def _write_dataset_yaml(path: Path, output_root: Path, class_name: str) -> None:
    content = "\n".join(
        [
            f"path: {output_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "nc: 1",
            "names:",
            f"  0: {class_name}",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def _split_for_index(index: int, train_count: int, val_count: int) -> str:
    if index < train_count:
        return "train"
    if index < (train_count + val_count):
        return "val"
    return "test"


def _density_bucket(count: int) -> str:
    if count <= 2:
        return "1-2"
    if count <= 8:
        return "3-8"
    return "9-30"


def _main() -> int:
    args = _parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be greater than 0")
    if args.cards_per_set <= 0:
        raise ValueError("--cards-per-set must be greater than 0")
    if args.img_size <= 64:
        raise ValueError("--img-size must be greater than 64")
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1")
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be between 0 and 1")
    if (args.train_ratio + args.val_ratio) >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    output_root = Path(args.output_root).resolve()
    paths = _prepare_output_dirs(output_root)

    excluded = {item.strip().lower() for item in str(args.exclude_set_ids).split(",") if item.strip()}
    set_ids = _load_set_ids(Path(args.metadata_path), excluded_set_ids=excluded)
    if not set_ids:
        raise ValueError("No set IDs found after applying exclusions")

    print(f"Loaded {len(set_ids)} set IDs from metadata")
    print(f"Excluded set IDs: {', '.join(sorted(excluded)) if excluded else '(none)'}")

    source_cards, source_summary = asyncio.run(
        _build_source_card_pool(
            set_ids=set_ids,
            cards_cache_dir=paths["cache_cards"],
            cards_per_set=args.cards_per_set,
            seed=args.seed,
            base_url=args.tcgdex_base_url,
            language=args.tcgdex_language,
            timeout=args.request_timeout,
            page_size=args.card_page_size,
            max_pages=args.max_set_pages,
        )
    )
    if not source_cards:
        raise RuntimeError("No source cards were downloaded from TCGdex")

    print(f"Source card pool size: {len(source_cards)}")
    if source_summary.get("warnings"):
        for warning in source_summary["warnings"]:
            print(f"WARN: {warning}")

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    train_count = int(args.samples * args.train_ratio)
    val_count = int(args.samples * args.val_ratio)
    test_count = max(0, args.samples - train_count - val_count)

    stats: dict[str, Any] = {
        "requested_samples": args.samples,
        "generated_samples": 0,
        "skipped_samples": 0,
        "split_counts": {"train": 0, "val": 0, "test": 0},
        "layout_counts": {"grid": 0, "free": 0},
        "density_bins": {"1-2": 0, "3-8": 0, "9-30": 0},
        "avg_cards_per_image": 0.0,
    }

    metadata_path = output_root / "samples.jsonl"
    max_attempts = max(args.samples * 3, args.samples + 50)

    total_cards_written = 0
    attempts = 0
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        while stats["generated_samples"] < args.samples and attempts < max_attempts:
            attempts += 1
            scene = _compose_scene(
                rng=rng,
                np_rng=np_rng,
                source_cards=source_cards,
                backgrounds_cache_dir=paths["cache_backgrounds"],
                img_size=args.img_size,
                min_bg_size=args.min_bg_size,
                max_bg_retries=args.max_bg_retries,
                timeout=args.request_timeout,
                bg_url_template=args.background_url_template,
                bg_min_index=args.background_min_index,
                bg_max_index=args.background_max_index,
                max_overlap_iou=args.max_overlap_iou,
                min_visible_ratio=args.min_visible_ratio,
                perspective_strength=args.perspective_strength,
                placement_attempts=args.placement_attempts,
                layout_grid_ratio=args.layout_grid_ratio,
            )

            if scene is None:
                stats["skipped_samples"] += 1
                continue

            generated_index = int(stats["generated_samples"])
            split = _split_for_index(generated_index, train_count=train_count, val_count=val_count)
            sample_id = f"sample_{generated_index + 1:06d}"

            image_path = paths[f"images_{split}"] / f"{sample_id}.jpg"
            label_path = paths[f"labels_{split}"] / f"{sample_id}.txt"

            cv2.imwrite(str(image_path), scene.image)
            label_path.write_text("\n".join(scene.labels) + "\n", encoding="utf-8")

            card_meta = [
                {
                    "set_id": item.source.set_id,
                    "card_id": item.source.card_id,
                    "source_path": str(item.source.local_path.relative_to(output_root)),
                    "visible_ratio": round(float(item.visible_ratio), 6),
                    "bbox_xyxy": [round(float(v), 3) for v in item.bbox],
                    "quad_xyxyxyxy": [round(float(v), 3) for v in item.quad.reshape(-1).tolist()],
                }
                for item in scene.cards
            ]
            sample_meta = {
                "sample_id": sample_id,
                "split": split,
                "layout": scene.layout_mode,
                "background_index": scene.background_index,
                "background_url": scene.background_url,
                "requested_card_count": scene.requested_count,
                "placed_card_count": len(scene.cards),
                "cards": card_meta,
            }
            metadata_file.write(json.dumps(sample_meta, ensure_ascii=True) + "\n")

            stats["generated_samples"] += 1
            stats["split_counts"][split] = int(stats["split_counts"][split]) + 1
            stats["layout_counts"][scene.layout_mode] = int(stats["layout_counts"][scene.layout_mode]) + 1
            stats["density_bins"][_density_bucket(len(scene.cards))] = int(stats["density_bins"][_density_bucket(len(scene.cards))]) + 1
            total_cards_written += len(scene.cards)

            if (generated_index + 1) % 250 == 0:
                print(f"Generated {generated_index + 1}/{args.samples} samples...")

    if stats["generated_samples"] == 0:
        raise RuntimeError("Dataset generation failed; no samples were produced")

    stats["avg_cards_per_image"] = float(total_cards_written / max(1, int(stats["generated_samples"])))

    dataset_yaml = output_root / "dataset.yaml"
    _write_dataset_yaml(dataset_yaml, output_root=output_root, class_name=args.class_name)

    source_manifest_path = output_root / "source_cards_manifest.json"
    source_manifest_path.write_text(json.dumps(source_summary, indent=2, ensure_ascii=True), encoding="utf-8")

    manifest = {
        "config": {
            "samples": args.samples,
            "img_size": args.img_size,
            "seed": args.seed,
            "cards_per_set": args.cards_per_set,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
            "layout_grid_ratio": args.layout_grid_ratio,
            "max_overlap_iou": args.max_overlap_iou,
            "min_visible_ratio": args.min_visible_ratio,
            "perspective_strength": args.perspective_strength,
            "background_url_template": args.background_url_template,
            "background_index_range": [args.background_min_index, args.background_max_index],
            "excluded_set_ids": sorted(excluded),
            "class_name": args.class_name,
        },
        "targets": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        },
        "stats": stats,
        "paths": {
            "dataset_yaml": str(dataset_yaml),
            "samples_jsonl": str(metadata_path),
            "source_manifest": str(source_manifest_path),
        },
    }

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    print("\nGeneration complete")
    print(f"Output root: {output_root}")
    print(f"Generated samples: {stats['generated_samples']} (requested {args.samples})")
    print(f"Skipped samples: {stats['skipped_samples']}")
    print(f"Split counts: {stats['split_counts']}")
    print(f"Density bins: {stats['density_bins']}")
    print(f"Average cards/image: {stats['avg_cards_per_image']:.2f}")
    print(f"Dataset YAML: {dataset_yaml}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
