import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
from PIL import Image
import sys

VIDEO_ID = 0
if len(sys.argv) > 1:
    VIDEO_ID = int(sys.argv[1])

CORNER_IDS = [0, 1, 2, 3]
WHITE_THRESHOLD = 200
MIN_HAND_AREA = 1500
DETECT_SCALE = 1.0
UPDATE_HZ = 30


def cv2glet(img, fmt):
    if fmt == 'GRAY':
        rows, cols = img.shape
        channels = 1
    else:
        rows, cols, channels = img.shape
    raw_img = Image.fromarray(img).tobytes()
    bytes_per_row = channels * cols
    return pyglet.image.ImageData(
        width=cols, height=rows, fmt=fmt,
        data=raw_img, pitch=-bytes_per_row,
    )


def warp_board(frame, corners, ids, out_w, out_h):
    if ids is None:
        return None
    ids_flat = ids.flatten().tolist()
    if not all(cid in ids_flat for cid in CORNER_IDS):
        return None

    centers = np.array([
        corners[ids_flat.index(cid)].reshape(-1, 2).mean(axis=0)
        for cid in CORNER_IDS
    ], dtype=np.float32)

    # Sort corners, code from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    s = centers.sum(axis=1)
    diff = np.diff(centers, axis=1).flatten()
    src_pts = np.zeros((4, 2), dtype=np.float32)
    src_pts[0] = centers[np.argmin(s)]
    src_pts[2] = centers[np.argmax(s)]
    src_pts[1] = centers[np.argmin(diff)]
    src_pts[3] = centers[np.argmax(diff)]

    dst_pts = np.array([
        [0,         0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0,         out_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(frame, M, (out_w, out_h))


def find_fingertip(board_bgr):
    gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    margin = px(40)
    mask[:margin, :] = 0
    mask[-margin:, :] = 0
    mask[:, :margin] = 0
    mask[:, -margin:] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_HAND_AREA:
        return None
    return tuple(largest[largest[:, :, 1].argmin()][0])


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

cap = cv2.VideoCapture(VIDEO_ID)
ret, first_frame = cap.read()
if not ret:
    print("Could not read from webcam.")
    sys.exit(1)
WINDOW_HEIGHT, WINDOW_WIDTH = first_frame.shape[:2]
print(f"Webcam resolution: {WINDOW_WIDTH} x {WINDOW_HEIGHT}")

SCALE = min(WINDOW_WIDTH, WINDOW_HEIGHT) / 720.0


def px(v):
    return max(1, int(round(v * SCALE)))


def draw_finger_overlay(board, fingertip):
    if fingertip is None:
        return
    cv2.circle(board, fingertip, px(20), (0, 255, 255), px(2), cv2.LINE_AA)
    cv2.circle(board, fingertip, px(4), (0, 0, 255), -1, cv2.LINE_AA)


def draw_border(board, color):
    cv2.rectangle(board, (0, 0), (WINDOW_WIDTH - 1, WINDOW_HEIGHT - 1),
                  color, px(8))


window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, caption="AR Game")
display_sprite = pyglet.sprite.Sprite(
    pyglet.image.ImageData(WINDOW_WIDTH, WINDOW_HEIGHT, 'BGR',
                           b'\x00' * (WINDOW_WIDTH * WINDOW_HEIGHT * 3),
                           pitch=-WINDOW_WIDTH * 3)
)
state = {
    'display': first_frame,
    'last_board': None,
}


def update_game(fingertip, dt):
    pass


def draw_game_overlay(board):
    pass


def update(dt):
    ret, frame = cap.read()
    if not ret:
        return

    small = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if corners:
        corners = [c / DETECT_SCALE for c in corners]

    board = warp_board(frame, corners, ids, WINDOW_WIDTH, WINDOW_HEIGHT)
    if board is not None:
        state['last_board'] = board
        display = cv2.flip(board.copy(), 1)  # Flip horizontally for more intuitive interaction
        fingertip = find_fingertip(display)
        draw_border(display, (0, 255, 0))
    elif state['last_board'] is not None:
        display = cv2.flip(state['last_board'].copy(), 1)
        fingertip = find_fingertip(display)
        draw_border(display, (0, 0, 255))
    else:
        display = cv2.flip(frame, 1)
        fingertip = None
        draw_border(display, (0, 0, 255))

    update_game(fingertip, dt)
    draw_game_overlay(display)
    draw_finger_overlay(display, fingertip)
    state['display'] = display

pyglet.clock.schedule_interval(update, 1 / UPDATE_HZ)


@window.event
def on_draw():
    window.clear()
    display_sprite.image = cv2glet(state['display'], 'BGR')
    display_sprite.draw()


@window.event
def on_close():
    cap.release()


pyglet.app.run()