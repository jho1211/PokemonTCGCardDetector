from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Disable expensive debug image writes for test runs.
os.environ.setdefault("DEBUG_SAVE_TRANSFORMS", "0")

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.identify import identify_card_from_image_bytes
from app.services.ocr import OCRFieldResult, get_ocr_service
from app.services.preprocess import (
    begin_debug_image_session,
    decode_image,
    detect_and_warp_card,
    end_debug_image_session,
    extract_regions,
    rotate_image_90,
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"[^A-Z0-9/]", "", value.upper())


def _contains_name(actual: str | None, expected: str | None) -> bool:
    if not expected:
        return True
    if not actual:
        return False
    return expected.strip().lower() in actual.strip().lower()


def _ocr_result_strength(result: OCRFieldResult, collector_field: bool) -> float:
    score = float(result.confidence)
    if result.text:
        score += 0.12
    if collector_field and result.text and "/" in result.text:
        score += 0.18
    if collector_field and result.text is None:
        score *= 0.5
    return score


def _orientation_score(number_result: OCRFieldResult, name_result: OCRFieldResult) -> float:
    number_signal = _ocr_result_strength(number_result, collector_field=True)
    name_signal = _ocr_result_strength(name_result, collector_field=False)
    return (0.75 * number_signal) + (0.25 * name_signal)


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_checks(checks: list[CheckResult]) -> None:
    for item in checks:
        mark = "PASS" if item.ok else "FAIL"
        print(f"[{mark}] {item.name}: {item.detail}")


def _response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    raise TypeError("Unsupported response object type")


async def _identify_with_timeout(image_bytes: bytes, timeout_seconds: float):
    return await asyncio.wait_for(identify_card_from_image_bytes(image_bytes), timeout=timeout_seconds)


def _run_local_ocr(
    card_image: Any,
    max_rotations: int,
    max_number_crops: int,
    max_name_crops: int,
) -> tuple[int, float, OCRFieldResult, OCRFieldResult]:
    ocr = get_ocr_service()
    best_turn = 0
    best_score = -1.0
    best_number = OCRFieldResult(text=None, raw_text=None, confidence=0.0)
    best_name = OCRFieldResult(text=None, raw_text=None, confidence=0.0)

    rotations = max(1, min(4, int(max_rotations)))
    number_crops = max(1, min(4, int(max_number_crops)))
    name_crops = max(1, min(3, int(max_name_crops)))

    for turns in range(rotations):
        oriented = rotate_image_90(card_image, turns)
        regions = extract_regions(oriented)

        number_result = ocr.extract_best_collector_number(regions["number"][:number_crops])
        name_result = ocr.extract_best_name(regions["name"][:name_crops])
        score = _orientation_score(number_result, name_result)

        if score > best_score:
            best_score = score
            best_turn = turns
            best_number = number_result
            best_name = name_result

    return best_turn, best_score, best_number, best_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fast CV/OCR pipeline checks on an input image.")
    parser.add_argument(
        "--image",
        default=str(BACKEND_ROOT / "app" / "images" / "example.tiff"),
        help="Path to image file to evaluate.",
    )
    parser.add_argument("--expect-number", default="147/165", help="Expected collector number.")
    parser.add_argument("--expect-name", default="Dratini", help="Expected card name (substring match).")
    parser.add_argument("--min-preprocess-score", type=float, default=0.50, help="Minimum acceptable preprocess score.")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="quick: preprocess+OCR only (fast). full: also run end-to-end identify pipeline.",
    )
    parser.add_argument(
        "--max-rotations",
        type=int,
        default=1,
        help="How many 90-degree rotations to evaluate for local OCR (1-4).",
    )
    parser.add_argument(
        "--max-number-crops",
        type=int,
        default=2,
        help="How many candidate number ROIs to evaluate per rotation (1-4).",
    )
    parser.add_argument(
        "--max-name-crops",
        type=int,
        default=1,
        help="How many candidate name ROIs to evaluate per rotation (1-3).",
    )
    parser.add_argument(
        "--pipeline-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for full mode end-to-end call.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="In full mode, require final identify-card response to succeed and match expected fields.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full identify-card JSON payload (only in full mode).",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 2

    t0 = time.perf_counter()
    image_bytes = image_path.read_bytes()

    debug_token = begin_debug_image_session("test_example")

    response_dict: dict[str, Any] | None = None
    full_error: str | None = None

    try:
        image = decode_image(image_bytes)
        preprocessed = detect_and_warp_card(image)

        ocr = get_ocr_service()
        best_turn = 0
        best_orientation_score = -1.0
        best_number = OCRFieldResult(text=None, raw_text=None, confidence=0.0)
        best_name = OCRFieldResult(text=None, raw_text=None, confidence=0.0)

        if ocr.is_available:
            best_turn, best_orientation_score, best_number, best_name = _run_local_ocr(
                card_image=preprocessed.image,
                max_rotations=args.max_rotations,
                max_number_crops=args.max_number_crops,
                max_name_crops=args.max_name_crops,
            )

        if args.mode == "full":
            try:
                response = asyncio.run(_identify_with_timeout(image_bytes, args.pipeline_timeout))
                response_dict = _response_to_dict(response)
            except TimeoutError:
                full_error = f"timed out after {args.pipeline_timeout:.1f}s"
            except Exception as exc:
                full_error = f"{type(exc).__name__}: {exc}"

    finally:
        end_debug_image_session(debug_token)

    local_checks: list[CheckResult] = [
        CheckResult(
            name="Card contour detected",
            ok=bool(preprocessed.detected_card),
            detail=f"detected_card={preprocessed.detected_card}",
        ),
        CheckResult(
            name="Preprocess score threshold",
            ok=preprocessed.score >= args.min_preprocess_score,
            detail=f"score={preprocessed.score:.3f}, min={args.min_preprocess_score:.3f}",
        ),
    ]

    if not ocr.is_available:
        local_checks.append(
            CheckResult(
                name="OCR availability",
                ok=False,
                detail=f"OCR unavailable: {ocr.last_init_error or 'unknown error'}",
            )
        )
    else:
        local_checks.extend(
            [
                CheckResult(
                    name="Collector number extracted",
                    ok=bool(best_number.text),
                    detail=(
                        f"number={best_number.text!r}, raw={best_number.raw_text!r}, "
                        f"conf={best_number.confidence:.3f}"
                    ),
                ),
                CheckResult(
                    name="Collector number matches expected",
                    ok=_normalize_text(best_number.text) == _normalize_text(args.expect_number),
                    detail=f"expected={args.expect_number!r}, actual={best_number.text!r}",
                ),
                CheckResult(
                    name="Name extracted",
                    ok=bool(best_name.text),
                    detail=f"name={best_name.text!r}, raw={best_name.raw_text!r}, conf={best_name.confidence:.3f}",
                ),
                CheckResult(
                    name="Name matches expected",
                    ok=_contains_name(best_name.text, args.expect_name),
                    detail=f"expected~={args.expect_name!r}, actual={best_name.text!r}",
                ),
                CheckResult(
                    name="Orientation selected",
                    ok=True,
                    detail=f"rotation={best_turn * 90} degrees, score={best_orientation_score:.3f}",
                ),
            ]
        )

    full_checks: list[CheckResult] = []

    if args.mode == "quick":
        full_checks.append(
            CheckResult(
                name="End-to-end identify call",
                ok=True,
                detail="skipped in quick mode (use --mode full to include API matching)",
            )
        )
    elif full_error is not None:
        full_checks.append(
            CheckResult(
                name="Pipeline response returned",
                ok=False,
                detail=full_error,
            )
        )
    else:
        full_checks.append(
            CheckResult(
                name="Pipeline response returned",
                ok=True,
                detail=f"success={response_dict.get('success')}, confidence={response_dict.get('confidence')}",
            )
        )

        if args.strict:
            cards = response_dict.get("cards") or []
            best_entry = {}
            if isinstance(cards, list) and cards:
                best_entry = next((item for item in cards if item.get("success")), cards[0])

            card = best_entry.get("card") if isinstance(best_entry, dict) else {}
            card = card if isinstance(card, dict) else {}
            full_checks.extend(
                [
                    CheckResult(
                        name="Strict: API identification succeeded",
                        ok=bool(response_dict.get("success")),
                        detail=f"success={response_dict.get('success')}, warning={response_dict.get('warning')}",
                    ),
                    CheckResult(
                        name="Strict: Final collector number matches",
                        ok=_normalize_text(str(card.get("collector_number", ""))) == _normalize_text(args.expect_number),
                        detail=f"expected={args.expect_number!r}, actual={card.get('collector_number')!r}",
                    ),
                    CheckResult(
                        name="Strict: Final name matches",
                        ok=_contains_name(card.get("name"), args.expect_name),
                        detail=f"expected~={args.expect_name!r}, actual={card.get('name')!r}",
                    ),
                ]
            )

    _print_section("Input")
    print(f"image={image_path}")
    print(f"mode={args.mode}")

    _print_section("Local CV/OCR Checks")
    _print_checks(local_checks)

    _print_section("End-to-End Pipeline Checks")
    _print_checks(full_checks)

    if args.json and response_dict is not None:
        _print_section("identify-card JSON")
        print(json.dumps(response_dict, indent=2, ensure_ascii=False))

    all_checks = local_checks + full_checks
    all_ok = all(item.ok for item in all_checks)

    elapsed = time.perf_counter() - t0
    _print_section("Summary")
    print(f"overall={'PASS' if all_ok else 'FAIL'}")
    print(f"elapsed_seconds={elapsed:.2f}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
