# Backend Pipeline Debug Guide

This guide explains how to visualize the Pokemon TCG Card Detection pipeline at each step using debug image output.

## Quick Start

### Run with Debug Image Saving

```bash
cd backend
python scripts/test_pipeline.py --debug
```

This will:
1. Execute the full backend pipeline on `test_cards.jpg`
2. Save detailed debug images at each processing step
3. Print the location of the debug images

### Output Location

Debug images are saved to:
```
backend/debug_outputs/{session_id}/
```

Where `{session_id}` is a unique identifier like `identify_1713693423_abc12345`

The test script will print the exact path when it completes.

## Understanding the Debug Images

The debug output shows the image at each step of the pipeline, in numbered order:

### Input & Detection
- **`00_input.png`** - Original input image
- **`10_warped_N.png`** - Card detected and warped by YOLO (N = card index)
  - Shows the perspective-corrected card after YOLO object detection

### Per-Card Analysis (one card = multiple stages)
- **`20_original_card.png`** - Original warped card before rotation attempts
- **`21_rotated_Xdeg.png`** - Card rotated by X degrees (0°, 90°, 180°, 270°)
  - Shows different orientations being tested for OCR
- **`22_rotation_N_result_...png`** - Result of rotation N with OCR results
  - Filename contains: score, extracted number text, extracted name text
- **`23_regions_rotation_Xdeg.png`** - Composite showing all extracted regions
  - Shows the number crop, name crop, and symbol crop regions in one image
  - Useful for understanding what text regions were extracted for OCR

## Pipeline Stages Explained

### Stage 00: Input
The raw image you provided to the pipeline.

### Stage 10: YOLO Detection & Warping
The YOLO model detects card-like objects and performs perspective correction to normalize the card into a flat, front-facing image. Multiple cards can be detected and warped.

### Stage 20-23: Per-Card Analysis
For each detected card:
1. **Stage 20**: Save the normalized card
2. **Stage 21**: Test different rotations (0°, 90°, 180°, 270°)
   - The pipeline tries multiple orientations because cards might be upside-down or sideways
3. **Stage 22**: For each rotation, attempt OCR to extract the collector number and name
   - The filename shows the confidence score and extracted text
4. **Stage 23**: Show the extracted region crops (number, name, symbol)
   - This shows what the OCR engine sees for each region

## Diagnosing Common Failures

### Problem: "Could not read collector number or card name"
**Look at**: `23_regions_rotation_*.png`

If the text crops are blurry, too small, or poorly positioned:
- Try a brighter photo
- Fill more of the frame with the card
- Avoid glare on the text areas

### Problem: Card detected but with low confidence
**Look at**: `21_rotated_*.png`

If the warped card looks skewed or poorly aligned:
- The card might not be flat against the camera
- Try a more straight-on angle
- Ensure the card edges are visible to the YOLO detector

### Problem: No cards detected at all
**Look at**: `00_input.png` and check if cards are clearly visible
- The YOLO model might need better lighting
- Ensure the card is fully in frame

## Advanced Usage

### Run with Custom Image
```bash
python scripts/test_pipeline.py --debug --image /path/to/your/image.jpg
```

### Run Multiple Orientation Tests
The default test only tries 1 rotation (for speed). To test all 4 orientations:
```bash
python scripts/test_pipeline.py --debug --max-rotations 4
```

### Save to Custom Debug Directory
Set the environment variable:
```bash
set DEBUG_IMAGE_DIR=d:\my_debug_output
python scripts/test_pipeline.py --debug
```

## Programmatic Access

To enable debug image saving in your own Python code:

```python
import os
os.environ["DEBUG_SAVE_TRANSFORMS"] = "1"  # Enable debug mode
os.environ["DEBUG_IMAGE_DIR"] = "path/to/debug/output"  # Optional

# Import after setting env vars
from app.services.identify import identify_card_from_image_bytes
import asyncio

image_bytes = open("my_card.jpg", "rb").read()
response = asyncio.run(identify_card_from_image_bytes(image_bytes))

# Debug images are now saved
print(f"Check debug_outputs/ for images")
```

## Image Interpretation Tips

### Viewing the Composite Region Image (`23_regions_*.png`)
- **Green text labels** show the region type (NUMBER, NAME, SYMBOL)
- Each crop is displayed in sequence
- If a crop is missing or black, that region couldn't be extracted

### Checking OCR Extraction (`22_rotation_*`)
- The filename embeds the extracted text
- `num_none` means no number was found
- `num_123/165` means the number "123/165" was successfully extracted

### Understanding Rotation (`21_rotated_*`)
- Compare the readability of text across different rotations
- The pipeline picks the rotation with the best OCR results
- You may see cards that are clearly upside-down in one rotation

## Performance Notes

- Debug image saving adds ~100-300ms per card depending on image size
- Storage: ~2-3 MB per complete debug session (5-10 PNG files)
- Disable with `DEBUG_SAVE_TRANSFORMS=0` or by removing the `--debug` flag

## Next Steps

If the debug images show clear regions but OCR is still failing:
- Check [OCR Configuration](../backend/app/services/ocr.py) for language/model settings
- Try different image preprocessing (brightness, contrast adjustment)
- Consider if the card is from a supported English set
