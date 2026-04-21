from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _coerce_batch(value: str) -> int | str:
    cleaned = str(value).strip().lower()
    if cleaned == "auto":
        return "auto"
    return int(cleaned)


def _coerce_cache(value: str) -> bool | str:
    cleaned = str(value).strip().lower()
    if cleaned in {"true", "1", "yes", "on"}:
        return True
    if cleaned in {"false", "0", "no", "off"}:
        return False
    if cleaned in {"ram", "disk"}:
        return cleaned
    raise ValueError("--cache must be one of: True, False, ram, disk")


def _main() -> int:
    data_path = Path("D:/Documents/GitHub/PokemonTCGCardDetector/backend/data/obb_synth/dataset.yaml").resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    from ultralytics import YOLO  # type: ignore[reportMissingImports]

    model = YOLO("D:/Documents/GitHub/PokemonTCGCardDetector/backend/data/yolo26n-obb.pt")
    result = model.train(data=data_path, epochs=100, imgsz=640, device=0, patience=5, batch=-1, cache="ram")

    print("\nTraining finished")
    save_dir = getattr(result, "save_dir", None)
    if save_dir:
        print(f"Run directory: {save_dir}")

    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        best_path = getattr(trainer, "best", None)
        if best_path:
            print(f"Best checkpoint: {best_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
