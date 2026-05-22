[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/5NorvP5a)


# Markers, Computer Vision, and AR

## Setup
1. Clone the repo and navigate to it via `cd assignment-04-cv-and-markers-Alphazerfall`.
2. Set it up a virtual enviroment running `python -m venv .venv`.
3. Activate the virtual environment using `.venv\Scripts\activate` on Windows and `source .venv/bin/activate` on Linux/Mac.
4. Install the required dependencies via `pip install -r requirements.txt`.

## Perspective Transformation
Navigate to the folder: `cd perspective_transformation`

With the venv activated, run:

```bash
python image_extractor.py INPUT_IMAGE OUTPUT_IMAGE [--width W --height H]
```

Example:

```bash
python image_extractor.py sample_image.jpg unwarped.jpg
```

**Arguments**
- `input_image` — path to the source image to load and display (e.g. `sample_image.jpg`). Any format OpenCV can read (JPEG, PNG, BMP, …) is fine.
- `output_image` — path where the warped result will be saved when you press `S`. The file extension determines the format (`.jpg`, `.png`, …).
- `--width W`, `--height H` — *optional.* Force a specific output resolution in pixels. Must be given together. If omitted, the output size is computed automatically from the selected points.

Click the four corners of the region you want to extract. After the last click the warped result appears in a second window.

**Controls**
- `ESC` — discard the selection and start over
- `S` — save the warped image to the output path (only available once the result is shown)
- `Q` — quit without saving


## AR Game
Navigate to the folder: `cd ar_game`

With the venv activated, run:

```bash
python AR_game.py [VIDEO_ID]
```

**Arguments**
- `VIDEO_ID` — *optional.* Index of the webcam to use (default `0`). On PCs with multiple cameras, try `1` if `0` is the wrong one.
