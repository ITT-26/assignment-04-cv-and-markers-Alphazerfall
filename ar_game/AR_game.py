import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
import threading
from PIL import Image
import sys


# ---------- Constants ----------
VIDEO_ID = 0
if len(sys.argv) > 1:
    VIDEO_ID = int(sys.argv[1])

# --- ArUco / board detection ---
CORNER_IDS = [0, 1, 2, 3]
DETECT_SCALE = 1.0
MAX_STALE_FRAMES = 15  # How many frames to keep showing the last board after losing detection

# --- Hand detection ---
HAND_DETECT_SCALE = 0.5
MIN_HAND_AREA = 1500

# --- App ---
UPDATE_HZ = 30
DEBUG_MASK = False  # Show mask used for hand detection


# ---------- Camera thread ----------
class CameraThread:
    def __init__(self, video_id):
        self.cap = cv2.VideoCapture(video_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._set_max_resolution()
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
 
    def _set_max_resolution(self):
        resolutions = [(1280, 720), (640, 480)]
        for w, h in resolutions:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (aw, ah) == (w, h):
                print(f"Camera resolution: {w}x{h}")
                return
 
    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
 
    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
 
    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()



# ---------- Setup ----------
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
aruco_params.adaptiveThreshWinSizeStep = 14
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

cap = CameraThread(VIDEO_ID)
first_frame = None
while first_frame is None:
    first_frame = cap.read()
WINDOW_HEIGHT, WINDOW_WIDTH = first_frame.shape[:2]

SCALE = min(WINDOW_WIDTH, WINDOW_HEIGHT) / 720.0

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, caption="AR Game")
display_sprite = pyglet.sprite.Sprite(
    pyglet.image.ImageData(WINDOW_WIDTH, WINDOW_HEIGHT, 'BGR',
                           b'\x00' * (WINDOW_WIDTH * WINDOW_HEIGHT * 3),
                           pitch=-WINDOW_WIDTH * 3)
)

state = {
    'display': first_frame,
    'last_board': None,
    'stale_frames': 0,
}


# ---------- Helpers ----------
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


def px(v):
    return max(1, int(round(v * SCALE)))


def zero_border(arr, m):
    arr[:m, :] = arr[-m:, :] = arr[:, :m] = arr[:, -m:] = 0


# ---------- Warping and Detection ----------
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


def detect_board(frame):
    small = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if corners:
        corners = [c / DETECT_SCALE for c in corners]
    return warp_board(frame, corners, ids, WINDOW_WIDTH, WINDOW_HEIGHT)


def find_fingertip(board_bgr):
    small = cv2.resize(board_bgr, None, fx=HAND_DETECT_SCALE, fy=HAND_DETECT_SCALE)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, blockSize=31, C=15)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    zero_border(mask, px(40 * HAND_DETECT_SCALE))

    if DEBUG_MASK:
        cv2.imshow("debug_mask", mask)
        cv2.waitKey(1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_HAND_AREA * HAND_DETECT_SCALE ** 2:
        return None

    pts = largest.reshape(-1, 2)
    bottom_y = pts[:, 1].max()
    bottom_pts = pts[pts[:, 1] >= bottom_y - 5]
    entry = bottom_pts.mean(axis=0)
    tip = pts[np.argmax(np.linalg.norm(pts - entry, axis=1))]

    inv = 1.0 / HAND_DETECT_SCALE
    return (int(tip[0] * inv), int(tip[1] * inv))


# ---------- Drawing ----------
def draw_finger_overlay(board, fingertip):
    if fingertip is None:
        return
    cv2.circle(board, fingertip, px(20), (0, 255, 255), px(2), cv2.LINE_AA)
    cv2.circle(board, fingertip, px(4), (0, 0, 255), -1, cv2.LINE_AA)


def draw_border(board, color):
    cv2.rectangle(board, (0, 0), (WINDOW_WIDTH - 1, WINDOW_HEIGHT - 1), color, px(8))


# ---------- Game hooks ----------
def update_game(fingertip, dt):
    pass


def draw_game_overlay(board):
    pass


# ---------- Main loop ----------
def update(dt):
    frame = cap.read()
    if frame is None:
        return

    board = detect_board(frame)

    if board is not None:
        state['last_board'] = board
        state['stale_frames'] = 0
        display = cv2.flip(board.copy(), 1)  # Flip horizontally for more intuitive interaction
        fingertip = find_fingertip(display)
        border_color = (0, 255, 0)  # Green border when board is detected
    elif state['last_board'] is not None and state['stale_frames'] < MAX_STALE_FRAMES:
        state['stale_frames'] += 1
        display = cv2.flip(state['last_board'].copy(), 1)
        fingertip = find_fingertip(display)
        border_color = (0, 165, 255)  # Orange border when showing stale board after losing detection
    else:
        display = cv2.flip(frame, 1)
        fingertip = None
        border_color = (0, 0, 255)  # Red border when no board is detected

    update_game(fingertip, dt)
    draw_game_overlay(display)
    draw_finger_overlay(display, fingertip)
    draw_border(display, border_color)
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