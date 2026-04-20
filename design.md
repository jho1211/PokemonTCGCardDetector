# Pokémon TCG Card Identification Backend — Design Document

## Goal

Build a backend service that accepts an image of a Pokémon trading card and identifies the exact card using:

1. OpenCV preprocessing + perspective correction
2. Set symbol identification using template matching
3. OCR of collector number (primary) and card name (fallback) using PaddleOCR
4. Querying the TCGdex API for final card resolution

This system should prioritize reliability, speed, and maintainability over training a large CV model.

---

# High-Level Flow

```text
User uploads image
→ Detect card boundaries
→ Perspective correction + crop
→ Extract set symbol region
→ Template match set symbol
→ OCR collector number
→ If needed: OCR card name
→ Query TCGdex API
→ Return best match + confidence score
```

---

# Tech Stack

## Backend

- Python 3.11+
- FastAPI

## Computer Vision

- OpenCV

## OCR

- PaddleOCR

## Card Database

- TCGdex REST API

Reference:
https://tcgdex.dev/rest/card

---

# Core Design Principles

## 1. Structured metadata first

Do NOT attempt full-image card classification.

Use:

- set symbol
- collector number
- card name (fallback)

These are far more reliable than artwork classification.

---

## 2. Collector number is the strongest identifier

Examples:

- 025/198
- TG14/TG30
- GG56/GG70
- SVP 045

The collector number should be treated as the primary lookup key.

---

## 3. Set identification should happen before OCR matching

The set symbol dramatically narrows the search space.

This improves:

- OCR accuracy tolerance
- API query precision
- overall confidence

---

# System Components

---

# 1. Image Upload Endpoint

## Endpoint

POST /identify-card

## Input

Multipart image upload:

- jpg
- jpeg
- png

## Output

```json
{
  "success": true,
  "card": {
    "id": "...",
    "name": "...",
    "set": "...",
    "number": "...",
    "image": "...",
    "rarity": "..."
  },
  "confidence": 0.94,
  "debug": {
    "set_match": "...",
    "ocr_number": "...",
    "ocr_name": "..."
  }
}
```

---

# 2. Card Detection + Perspective Correction

## Goal

Locate the physical card in the uploaded image and transform it into a clean, front-facing crop.

## Approach

Use OpenCV:

1. Convert to grayscale
2. Gaussian blur
3. Edge detection (Canny)
4. Find contours
5. Detect largest 4-point contour
6. Apply perspective transform
7. Normalize output dimensions

## Notes

Expected output:

- upright card crop
- consistent dimensions
- minimal background

This should avoid needing object detection models initially.

---

# 3. Region Extraction

## Goal

Extract fixed regions for:

- set symbol
- collector number
- card name

## Assumption

After perspective correction, the card dimensions are standardized.

This allows fixed coordinate crops.

## Required Regions

### A. Set Symbol Region

Usually bottom-right area of the card.

### B. Collector Number Region

Usually bottom-left / bottom-right depending on set layout.

### C. Card Name Region

Top-left / top-center area.

Used only if collector number lookup is ambiguous.

---

# 4. Set Symbol Identification

## Method

Template matching using OpenCV.

## Approach

Maintain a local template library:

```text
/templates/
    base_set.png
    jungle.png
    fossil.png
    ...
```

Each template should include:

- clean symbol image
- associated TCGdex set code

## Matching Strategy

Use:

`cv2.matchTemplate()`

Compare extracted symbol region against all templates.

Return:

- best match
- similarity score

## Confidence Threshold

Define minimum threshold.

If below threshold:

- mark set as uncertain
- continue with OCR-first fallback path

---

# 5. OCR: Collector Number

## Primary Identifier

Collector number is the main lookup key.

## Tool

PaddleOCR

## Input

Only the cropped collector number region.

## Expected Output

Examples:

- 025/198
- 034/182
- TG14/TG30
- GG56/GG70

## Post-processing

Normalize OCR output:

- remove whitespace
- standardize slashes
- uppercase letters
- handle common OCR mistakes

Examples:

- O → 0
- I → 1
- S → 5 (carefully)

This normalization is critical.

---

# 6. OCR: Card Name (Fallback)

## Use only when:

- collector number OCR fails
- multiple candidates are returned
- set match confidence is low

## Tool

PaddleOCR

## Input

Card name crop

## Matching

Use fuzzy string matching against TCGdex candidates.

This should NOT be the primary identifier.

---

# 7. TCGdex Resolution

## Primary Query Path

Use:

- set code
- collector number

to resolve exact match.

## Secondary Query Path

Use:

- set code
- fuzzy card name

when collector number is unavailable.

## Final Validation

Compare:

- OCR name
- API result name

to improve confidence score.

---

# Confidence Scoring

## Example weighted score

```text
Set symbol match:       40%
Collector number OCR:   40%
Card name validation:   20%
```

Final score should determine:

- auto-accept
- low-confidence warning
- manual review suggestion

---

# Debug Mode

Store intermediate outputs for development:

- corrected card image
- cropped symbol image
- OCR regions
- template match scores
- raw OCR output

This is extremely important during iteration.

---

# Suggested Project Structure

```text
backend/
    app/
        main.py
        api/
            routes.py
        services/
            preprocess.py
            template_match.py
            ocr.py
            tcgdex.py
            matcher.py
        templates/
            set_symbols/
        utils/
            image.py
            normalization.py
```

---

# MVP Scope

Version 1 should support:

- English cards only
- modern standard layout cards only
- front-facing photos only
- single card per image only

Do NOT start with:

- Japanese cards
- slabs
- angled stacks
- multiple cards
- vintage edge cases

Start narrow.

Expand later.

---

# Open Questions / Design Gaps

These should be decided before implementation:

## 1. Which card eras are supported first?

Modern Scarlet/Violet only?

Sword/Shield?

All eras?

This affects region extraction heavily.

---

## 2. Will set symbol templates be manually collected or auto-generated?

Template quality matters significantly.

This should be standardized.

---

## 3. Should API results be cached locally?

Frequent TCGdex queries may benefit from local caching.

Likely recommended.

---

## 4. Should failed identifications be stored for future review?

Very useful for improving OCR normalization and templates.

Likely recommended.

---

# Recommended First Milestone

Implement only:

```text
Perspective correction
+
Collector number OCR
+
Manual test lookup
```

before building full template matching.

This will validate the highest-value part of the system first.