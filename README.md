[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/5NorvP5a)

# Markers, Computer Vision, and AR

## Setup
1. Clone the repo and navigate to it via `cd assignment-04-cv-and-markers-Alphazerfall`.
2. Set up a virtual environment by running `python -m venv .venv`.
3. Activate the virtual environment using `.venv\Scripts\activate` on Windows and `source .venv/bin/activate` on Linux/Mac.
4. Install the required dependencies via `pip install -r requirements.txt`.

## Perspective Transformation
Navigate to the folder: `cd perspective_transformation`

Run [`image_extractor.py`](./perspective_transformation/image_extractor.py) via the command line:

```bash
python image_extractor.py -i INPUT_IMAGE -o OUTPUT_IMAGE [--width W --height H]
```

**Arguments**

| flag | parameter | required? | default |
|------|-----------|-----------|---------|
| `-i`, `--input`  | path to the input image (e.g. `sample_image.jpg`). Any format OpenCV can read (JPEG, PNG, BMP, …) is fine. | yes | – |
| `-o`, `--output` | path where the warped result will be saved when you press `S`. The file extension determines the format. If the file already exists, `_1`, `_2`, … is appended so nothing is overwritten. | yes | – |
| `--width`        | width of the warped output image in pixels. Must be used together with `--height`. | no | auto-computed from the selected points |
| `--height`       | height of the warped output image in pixels. Must be used together with `--width`. | no | auto-computed from the selected points |

**Examples**

Minimum parameters (output size derived from the selected points):
```bash
python image_extractor.py -i sample_image.jpg -o unwarped.jpg
```

With a forced output resolution:
```bash
python image_extractor.py -i sample_image.jpg -o unwarped.jpg --width 750 --height 500
```

**Usage**

- The input image opens in the *Selection Window*. Click four corners of the region you want to extract — the selected points and connecting lines are drawn as you go.
- After the fourth click the *Result Window* opens with the warped image.
- Press `S` in either window to save the warped image to the output path.
- Press `ESC` to discard the current selection (and close the result window) so you can start over.
- Press `Q` to quit without saving.

## AR Game
Navigate to the folder: `cd ar_game`

Run [`AR_game.py`](./ar_game/AR_game.py) via the command line:

```bash
python AR_game.py [VIDEO_ID]
```

**Arguments**

| parameter | required? | default |
|-----------|-----------|---------|
| `VIDEO_ID` | index of the webcam to use. On PCs with multiple cameras, try `1` if `0` is the wrong one. | no | `0` |


**Setup**

Print four ArUco markers from the `DICT_6X6_250` dictionary with IDs `0`, `1`, `2`, `3` and place them on the corners of a flat surface: top-left = `0`, top-right = `1`, bottom-right = `2`, bottom-left = `3`. Point your webcam at the board so all four markers are visible.

**Gameplay**

The region between the four markers is detected, perspective-warped to fill the window, and mirrored. Reach into the board with one pointing finger — a red dot marks the detected fingertip.

- On the menu screen, hold your finger over the **START** button to begin a round.
- During the round, hold your finger on a mole for a few frames to whack it. Score as many hits as you can in 60 seconds.
- Moles fade in when they appear and fade out as their time runs out, so you can see how long you have to react.
- On the result screen, hold your finger over **PLAY AGAIN** for another round.

**Border colors**

The border around the window indicates the tracking state:

| Color  | Meaning |
|--------|---------|
| Green  | All four markers detected: tracking is live. |
| Orange | Markers temporarily lost: the last known board is shown for a short grace period so brief occlusions don't reset the game. |
| Red    | No board available: move the camera so all four markers are visible again. |

**Keyboard shortcuts**

| Key | Action |
|-----|--------|
| `R` | Restart the current round. |
| `Q` | Quit the game. |