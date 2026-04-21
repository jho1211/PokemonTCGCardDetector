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

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Parse arguments early, before imports, to set environment variables
_early_parser = argparse.ArgumentParser(add_help=False)
_early_parser.add_argument("--debug", action="store_true")
_early_args, _remaining = _early_parser.parse_known_args()

# Set environment variables based on early args before any imports
if _early_args.debug:
    os.environ["DEBUG_SAVE_TRANSFORMS"] = "1"
else:
    os.environ.setdefault("DEBUG_SAVE_TRANSFORMS", "0")

# Now import app modules with correct environment variables
from app.services.identify import identify_card_from_image_bytes
from app.config.config import (
    OCR_COLLECTOR_EARLY_STOP_SCORE,
    OCR_NAME_EARLY_STOP_SCORE,
    OCR_ORIENTATION_EARLY_STOP_SCORE,
    OCR_ROTATION_PRIORITY,
)
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

    for turns in OCR_ROTATION_PRIORITY[:rotations]:
        oriented = rotate_image_90(card_image, turns)
        regions = extract_regions(oriented)

        number_result = ocr.extract_best_collector_number(
            regions["number"][:number_crops],
            stop_score=OCR_COLLECTOR_EARLY_STOP_SCORE,
        )
        name_result = ocr.extract_best_name(
            regions["name"][:name_crops],
            stop_score=OCR_NAME_EARLY_STOP_SCORE,
        )
        score = _orientation_score(number_result, name_result)

        if score > best_score:
            best_score = score
            best_turn = turns
            best_number = number_result
            best_name = name_result

        if (
            number_result.text
            and name_result.text
            and number_result.confidence >= OCR_COLLECTOR_EARLY_STOP_SCORE
            and name_result.confidence >= OCR_NAME_EARLY_STOP_SCORE
        ):
            break
        if number_result.text and name_result.text and score >= OCR_ORIENTATION_EARLY_STOP_SCORE:
            break

    return best_turn, best_score, best_number, best_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full backend pipeline test on an input image with strict card name validation.")
    parser.add_argument(
        "--image",
        default=str(BACKEND_ROOT / "data" / "test_cards.jpg"),
        help="Path to image file to evaluate. Default: test_cards.jpg from data/",
    )
    # Note: --debug and --expected-names are parsed early before imports above
    parser.add_argument(
        "--min-preprocess-score", type=float, default=0.50, help="Minimum acceptable preprocess score."
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="full",
        help="quick: preprocess+OCR only (fast). full: run end-to-end identify pipeline (default).",
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
        "--json-output",
        default=str(BACKEND_ROOT / "scripts" / "test_cards_output.json"),
        help="Path to save full IdentifyCardResponse JSON. Default: scripts/test_cards_output.json",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug image saving at each pipeline step. Images saved to debug_outputs/{session_id}/",
    )
    parser.add_argument(
        "--expected-names",
        action="append",
        help="Expected card names (repeatable, case-insensitive substring match). Default: Pikachu, Dratini",
    )
    args = parser.parse_args()

    # Handle expected names - use provided values or defaults
    expected_names_list = args.expected_names if args.expected_names else ["Pikachu", "Dratini"]
    expected_names_lower = set(name.strip().lower() for name in expected_names_list if name.strip())

    # Note: DEBUG_SAVE_TRANSFORMS was already set based on _early_args.debug before imports
    if _early_args.debug:
        debug_output_dir = BACKEND_ROOT / "debug_outputs"
        print(f"Debug mode enabled. Images will be saved to: {debug_output_dir}/<session_id>/")

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return 2

    json_output_path = Path(args.json_output).expanduser().resolve()
    json_output_path.parent.mkdir(parents=True, exist_ok=True)

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
                    name="OCR backend invocation",
                    ok=True,
                    detail=f"backend={ocr.last_call_backend!r}, result_format={ocr.last_result_format!r}",
                ),
                CheckResult(
                    name="OCR runtime error",
                    ok=ocr.last_runtime_error is None,
                    detail=ocr.last_runtime_error or "none",
                ),
                CheckResult(
                    name="Collector number extracted",
                    ok=bool(best_number.text),
                    detail=(
                        f"number={best_number.text!r}, raw={best_number.raw_text!r}, "
                        f"conf={best_number.confidence:.3f}"
                    ),
                ),
                CheckResult(
                    name="Name extracted",
                    ok=bool(best_name.text),
                    detail=f"name={best_name.text!r}, raw={best_name.raw_text!r}, conf={best_name.confidence:.3f}",
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

        # Extract detected card names for validation
        cards = response_dict.get("cards") or []
        detected_names = set()
        
        if isinstance(cards, list):
            for card_entry in cards:
                if isinstance(card_entry, dict):
                    card_payload = card_entry.get("card")
                    if isinstance(card_payload, dict) and card_payload.get("name"):
                        detected_names.add(card_payload.get("name").strip().lower())
        
        # Check which expected names were found
        missing_names = expected_names_lower - detected_names
        found_names = expected_names_lower & detected_names
        
        # Strict validation: all expected names must be found
        all_expected_found = len(missing_names) == 0
        
        full_checks.extend([
            CheckResult(
                name="Expected card names detected",
                ok=all_expected_found,
                detail=f"expected={sorted(expected_names_lower)}, found={sorted(found_names)}, missing={sorted(missing_names) if missing_names else 'none'}",
            ),
        ])

    _print_section("Input")
    print(f"image={image_path}")
    print(f"mode={args.mode}")
    print(f"expected_names={sorted(expected_names_lower)}")

    _print_section("Local CV/OCR Checks")
    _print_checks(local_checks)

    _print_section("End-to-End Pipeline Checks")
    _print_checks(full_checks)

    if response_dict is not None:
        # Save full JSON response
        try:
            with open(json_output_path, "w") as f:
                json.dump(response_dict, f, indent=2, ensure_ascii=False)
            print(f"\nJSON output saved to: {json_output_path}")
        except Exception as e:
            print(f"\nWarning: Failed to save JSON output: {e}")

    if _early_args.debug:
        debug_output_dir = BACKEND_ROOT / "debug_outputs"
        if debug_output_dir.exists():
            # Find the most recent debug session directory
            session_dirs = sorted([d for d in debug_output_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
            if session_dirs:
                latest_session = session_dirs[0]
                print(f"\nDebug images saved to: {latest_session}")
                image_count = len(list(latest_session.glob("*.png")))
                print(f"Generated {image_count} debug images")
        print("\nDebug images at each step:")
        print("  00_input - original input image")
        print("  10_warped_N - YOLO-detected and warped card N")
        print("  20_original_card - original warped card before rotation")
        print("  21_rotated_Xdeg - card rotated by X degrees")
        print("  22_rotation_N_result_... - result image with OCR results in filename")
        print("  23_regions_rotation_Xdeg - composite showing number/name/symbol crops")

    all_checks = local_checks + full_checks
    all_ok = all(item.ok for item in all_checks)

    elapsed = time.perf_counter() - t0
    _print_section("Summary")
    print(f"overall={'PASS' if all_ok else 'FAIL'}")
    print(f"elapsed_seconds={elapsed:.2f}")
    if response_dict and "cards" in response_dict:
        print(f"cards_detected={len(response_dict.get('cards', []))}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
