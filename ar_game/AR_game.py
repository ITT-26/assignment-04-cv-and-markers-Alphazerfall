import os
import sys

# Disabling the Windows Media Foundation hardware transforms allows the Logitech C920 to properly deliver MJPG frames at 1080p
# Must be set before cv2 is imported to take effect on Windows (MSMF backend). Harmless on other platforms.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
import random
import threading


# ---------- Constants ----------
VIDEO_ID = 0
if len(sys.argv) > 1:
    VIDEO_ID = int(sys.argv[1])

# --- ArUco / board detection ---
CORNER_IDS = [0, 1, 2, 3]
DETECT_SCALE = 1.0
MAX_STALE_FRAMES = 15  # how many frames to keep showing the last board after losing detection

# --- Hand detection ---
HAND_DETECT_MAX_W = 640  # cap finger-detection input width regardless of camera resolution
ZERO_BORDER_PX = 45  # border area to ignore for hand detection, as it often contains noise
MIN_HAND_AREA = 1500

# --- App ---
UPDATE_HZ = 30
DEBUG_MASK = True  # show mask used for hand detection in a separate window

# --- Game ---
GAME_DURATION = 60.0  # seconds per round
GRID_COLS = 3
GRID_ROWS = 3
MOLE_RADIUS = 65
MOLE_SHOW_TIME = 1.6
MOLE_MIN_GAP = 0.4
MOLE_MAX_GAP = 1.1
MAX_ACTIVE_MOLES = 2
HIT_HOVER_FRAMES = 3

# Color Palette (BGR)
COL_BG_PANEL    = (40, 40, 40)
COL_ACCENT      = (90, 200, 255)  # warm amber
COL_TEXT        = (255, 255, 255)
COL_TEXT_DARK   = (30, 30, 30)
COL_MOLE_BODY   = (60, 100, 170)
COL_MOLE_RIM    = (35, 60, 110)
COL_MOLE_HIT    = (80, 220, 120)
COL_HOVER_RING  = (90, 200, 255)
COL_BUTTON      = (90, 200, 255)
COL_BUTTON_TXT  = (20, 20, 20)


# ---------- Camera thread ----------
class CameraThread:
    def __init__(self, video_id):
        backend = cv2.CAP_MSMF if sys.platform == 'win32' else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(video_id, backend)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._set_max_resolution()
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _set_max_resolution(self):
        resolutions = [(1920, 1080), (1280, 720), (640, 480)]
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


# ---------- Helpers ----------
def px(v):
    return max(1, int(round(v * SCALE)))


def zero_border(arr, m):
    arr[:m, :] = arr[-m:, :] = arr[:, :m] = arr[:, -m:] = 0


def point_in_rect(pt, rect):
    if pt is None:
        return False
    x, y = pt
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


# ---------- Vision ----------
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
    h, w = board_bgr.shape[:2]
    scale = min(1.0, HAND_DETECT_MAX_W / w)
    small = cv2.resize(board_bgr, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=91, C=15)

    # Small kernel opening to remove white specks, then large closing to close hand blobs
    open_kernel  = np.ones((5, 5), np.uint8)
    close_kernel = np.ones((21, 21), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    # Fill contour interiors to get solid blobs
    ext, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, ext, -1, 255, cv2.FILLED)

    zero_border(mask, px(ZERO_BORDER_PX * scale))

    if DEBUG_MASK:
        cv2.imshow("debug_mask", mask)
        cv2.waitKey(1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Find highest point across all hand-sized blobs so a small fingertip beats a larger palm blob
    significant = [c for c in contours if cv2.contourArea(c) >= MIN_HAND_AREA * scale ** 2]
    if not significant:
        return None

    # Get the single highest point among all significant contours
    all_pts = np.vstack(significant).reshape(-1, 2)
    tip = all_pts[np.argmin(all_pts[:, 1])]

    inv = 1.0 / scale
    return (int(tip[0] * inv), int(tip[1] * inv))


# ---------- Drawing ----------
def draw_border(board, color):
    cv2.rectangle(board, (0, 0), (WINDOW_WIDTH - 1, WINDOW_HEIGHT - 1), color, px(8))


def draw_finger_overlay(board, fingertip):
    if fingertip is None:
        return
    cv2.circle(board, fingertip, px(20), COL_ACCENT, px(2), cv2.LINE_AA)
    cv2.circle(board, fingertip, px(4), (0, 0, 255), -1, cv2.LINE_AA)


def draw_circle_alpha(board, center, radius, color, alpha, thickness=-1):
    if alpha <= 0:
        return
    if alpha >= 1.0:
        cv2.circle(board, center, radius, color, thickness, cv2.LINE_AA)
        return
    cx, cy = center
    extra = thickness if thickness > 0 else 0
    pad = radius + extra + 4
    H, W = board.shape[:2]
    x1, y1 = max(0, cx - pad), max(0, cy - pad)
    x2, y2 = min(W, cx + pad), min(H, cy + pad)
    if x2 <= x1 or y2 <= y1:
        return
    roi = board[y1:y2, x1:x2]
    overlay = roi.copy()
    cv2.circle(overlay, (cx - x1, cy - y1), radius, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def draw_text_centered(board, text, center, font_scale, color, thickness, outline=True):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    x = int(center[0] - w / 2)
    y = int(center[1] + h / 2)
    if outline:
        cv2.putText(board, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                    font_scale, COL_TEXT_DARK, thickness + px(3), cv2.LINE_AA)
    cv2.putText(board, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def draw_panel(board, top_left, bottom_right, color=COL_BG_PANEL, alpha=0.75):
    np.copyto(_overlay_buf, board)
    cv2.rectangle(_overlay_buf, top_left, bottom_right, color, -1, cv2.LINE_AA)
    cv2.addWeighted(_overlay_buf, alpha, board, 1 - alpha, 0, board)


def draw_button(board, rect, label, hover_progress):
    x1, y1, x2, y2 = rect
    draw_panel(board, (x1, y1), (x2, y2), COL_BUTTON, alpha=0.85)
    cv2.rectangle(board, (x1, y1), (x2, y2), COL_TEXT_DARK, px(3), cv2.LINE_AA)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    draw_text_centered(board, label, (cx, cy), SCALE * 1.1, COL_BUTTON_TXT, px(2), outline=False)

    if hover_progress > 0:
        cv2.rectangle(board, (x1, y1), (x2, y2), COL_MOLE_HIT, px(5), cv2.LINE_AA)
        bar_w = int((x2 - x1) * hover_progress)
        cv2.rectangle(board, (x1, y2 - px(6)), (x1 + bar_w, y2), COL_MOLE_HIT, -1, cv2.LINE_AA)


def draw_waiting_screen(board):
    draw_text_centered(board, "SHOW THE BOARD TO THE CAMERA",
                       (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - px(20)),
                       SCALE * 1.2, COL_ACCENT, px(3))
    draw_text_centered(board, "Make sure all 4 corner markers are visible",
                       (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + px(40)),
                       SCALE * 0.6, COL_TEXT, px(1))


# ---------- Game ----------
SCREEN_MENU, SCREEN_PLAYING, SCREEN_RESULT = 'menu', 'playing', 'result'


class WhackAMoleGame:
    BUTTON_HOLD_FRAMES = HIT_HOVER_FRAMES * 4

    def __init__(self):
        self.cells = self._grid_positions()
        self._reset()
        self.screen = SCREEN_MENU

    # ----- Layout -----
    @staticmethod
    def _grid_positions():
        positions = []
        margin_x = WINDOW_WIDTH  / (GRID_COLS + 1)
        margin_y = WINDOW_HEIGHT / (GRID_ROWS + 1)
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                x = int(margin_x * (col + 1))
                y = int(margin_y * (row + 1))
                positions.append((x, y))
        return positions

    @staticmethod
    def _start_button_rect():
        bw, bh = px(280), px(90)
        cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + px(40)
        return (cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2)

    @staticmethod
    def _restart_button_rect():
        bw, bh = px(280), px(90)
        cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + px(80)
        return (cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2)

    def _reset(self):
        self.moles = []
        self.spawn_timer = MOLE_MIN_GAP
        self.score = 0
        self.time_left = GAME_DURATION
        self.button_hover_frames = 0
        self.tracking_active = True

    def start_round(self):
        self._reset()
        self.screen = SCREEN_PLAYING

    # ----- Update -----
    def update(self, fingertip, dt, tracking_active):
        self.tracking_active = tracking_active

        if self.screen == SCREEN_MENU:
            self._update_button(fingertip, self._start_button_rect())
        elif self.screen == SCREEN_RESULT:
            self._update_button(fingertip, self._restart_button_rect())
        else:
            self._update_playing(fingertip, dt)

    def _update_button(self, fingertip, rect):
        if self.tracking_active and point_in_rect(fingertip, rect):
            self.button_hover_frames += 1
            if self.button_hover_frames >= self.BUTTON_HOLD_FRAMES:
                self.start_round()
        else:
            self.button_hover_frames = 0

    def _update_playing(self, fingertip, dt):
        self.time_left -= dt
        if self.time_left <= 0:
            self.time_left = 0
            self.screen = SCREEN_RESULT
            self.button_hover_frames = 0
            self.moles = []
            return

        live_moles = [m for m in self.moles if not m['hit']]
        self.spawn_timer -= dt
        if self.spawn_timer <= 0 and len(live_moles) < MAX_ACTIVE_MOLES:
            self._spawn_mole()
            self.spawn_timer = random.uniform(MOLE_MIN_GAP, MOLE_MAX_GAP)

        radius_px = px(MOLE_RADIUS)

        for m in self.moles:
            if m['hit']:
                m['hit_age'] += dt
                continue

            m['age'] += dt

            if self.tracking_active and fingertip is not None:
                dx = fingertip[0] - m['x']
                dy = fingertip[1] - m['y']
                if dx * dx + dy * dy <= radius_px * radius_px:
                    m['hover_frames'] += 1
                    if m['hover_frames'] >= HIT_HOVER_FRAMES:
                        m['hit'] = True
                        self.score += 1
                else:
                    m['hover_frames'] = 0
            else:
                m['hover_frames'] = 0

        self.moles = [
            m for m in self.moles
            if (m['hit'] and m['hit_age'] < 0.3)
            or (not m['hit'] and m['age'] < MOLE_SHOW_TIME)
        ]

    def _spawn_mole(self):
        occupied = {(m['x'], m['y']) for m in self.moles}
        free = [c for c in self.cells if c not in occupied]
        if not free:
            return
        x, y = random.choice(free)
        self.moles.append({
            'x': x, 'y': y, 'age': 0.0,
            'hit': False, 'hit_age': 0.0, 'hover_frames': 0,
        })

    # ----- Draw -----
    def draw(self, board):
        if self.screen == SCREEN_MENU:
            self._draw_menu(board)
        elif self.screen == SCREEN_PLAYING:
            self._draw_playing(board)
        else:
            self._draw_result(board)

    def _draw_menu(self, board):
        draw_text_centered(board, "WHACK-A-MOLE",
                           (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - px(120)),
                           SCALE * 2.2, COL_ACCENT, px(3))
        draw_text_centered(board, f"Hit as many moles as you can in {int(GAME_DURATION)} seconds",
                           (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - px(40)),
                           SCALE * 0.7, COL_TEXT, px(1))

        rect = self._start_button_rect()
        progress = self.button_hover_frames / self.BUTTON_HOLD_FRAMES
        draw_button(board, rect, "START", min(progress, 1.0))

        draw_text_centered(board, "Hold finger on START to begin   |   Q: quit",
                           (WINDOW_WIDTH // 2, WINDOW_HEIGHT - px(40)),
                           SCALE * 0.55, COL_TEXT, px(1))

    def _draw_playing(self, board):
        radius_px = px(MOLE_RADIUS)

        for m in self.moles:
            cx, cy = m['x'], m['y']

            if m['hit']:
                grow = 1.0 + m['hit_age'] / 0.3 * 0.7
                cv2.circle(board, (cx, cy), int(radius_px * grow),
                           COL_MOLE_HIT, px(5), cv2.LINE_AA)
            else:
                life_pct = m['age'] / MOLE_SHOW_TIME
                if life_pct < 0.08:  # quick fade-in
                    alpha = life_pct / 0.08
                elif life_pct > 0.6:  # fade-out over last 40%
                    alpha = max(0.0, (1.0 - life_pct) / 0.4)
                else:
                    alpha = 1.0

                draw_circle_alpha(board, (cx + px(4), cy + px(6)), radius_px,
                                  (20, 20, 20), alpha * 0.7)
                draw_circle_alpha(board, (cx, cy), radius_px, COL_MOLE_BODY, alpha)
                draw_circle_alpha(board, (cx, cy), radius_px, COL_MOLE_RIM, alpha, px(3))

                if m['hover_frames'] > 0:
                    pct = min(1.0, m['hover_frames'] / HIT_HOVER_FRAMES)
                    cv2.ellipse(board, (cx, cy),
                                (radius_px + px(10), radius_px + px(10)),
                                -90, 0, int(360 * pct),
                                COL_HOVER_RING, px(4), cv2.LINE_AA)

        # HUD bar at the top
        bar_h = px(60)
        draw_panel(board, (0, 0), (WINDOW_WIDTH, bar_h), COL_BG_PANEL, alpha=0.75)

        draw_text_centered(board, f"Score  {self.score}",
                           (px(140), bar_h // 2), SCALE * 0.95,
                           COL_TEXT, px(2), outline=False)
        draw_text_centered(board, f"Time  {int(self.time_left)}s",
                           (WINDOW_WIDTH - px(140), bar_h // 2),
                           SCALE * 0.95, COL_ACCENT, px(2), outline=False)

        pct = self.time_left / GAME_DURATION
        cv2.rectangle(board, (0, bar_h), (int(WINDOW_WIDTH * pct), bar_h + px(6)),
                      COL_ACCENT, -1, cv2.LINE_AA)

    def _draw_result(self, board):
        draw_panel(board, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 0), alpha=0.6)

        draw_text_centered(board, "TIME'S UP",
                           (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - px(160)),
                           SCALE * 1.8, COL_ACCENT, px(3))
        draw_text_centered(board, f"Score: {self.score}",
                           (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - px(40)),
                           SCALE * 2.0, COL_TEXT, px(3))

        rect = self._restart_button_rect()
        progress = self.button_hover_frames / self.BUTTON_HOLD_FRAMES
        draw_button(board, rect, "PLAY AGAIN", min(progress, 1.0))

        draw_text_centered(board, "Hold finger on PLAY AGAIN  |  R: restart  |  Q: quit",
                           (WINDOW_WIDTH // 2, WINDOW_HEIGHT - px(40)),
                           SCALE * 0.55, COL_TEXT, px(1))


# ---------- Main loop ----------
def update(dt):
    frame = cap.read()
    if frame is None:
        return

    board = detect_board(frame)

    if board is not None:
        state['last_board'] = board
        state['stale_frames'] = 0
        state['board_ever_detected'] = True
        display = cv2.flip(board.copy(), 1)
        fingertip = find_fingertip(display)
        border_color = (0, 255, 0)
        tracking_active = True
    elif state['last_board'] is not None and state['stale_frames'] < MAX_STALE_FRAMES:
        state['stale_frames'] += 1
        display = cv2.flip(state['last_board'].copy(), 1)
        fingertip = find_fingertip(display)
        border_color = (0, 165, 255)
        tracking_active = False
    else:
        display = cv2.flip(frame, 1)
        fingertip = None
        border_color = (0, 0, 255)
        tracking_active = False

    if state['board_ever_detected']:
        game.update(fingertip, dt, tracking_active)
        game.draw(display)
    else:
        draw_waiting_screen(display)

    draw_finger_overlay(display, fingertip)
    draw_border(display, border_color)
    state['display'] = display

    # Update the texture for display
    img = pyglet.image.ImageData(WINDOW_WIDTH, WINDOW_HEIGHT, 'BGR',
                                  display.tobytes(), pitch=-WINDOW_WIDTH * 3)
    _display_tex.blit_into(img, 0, 0, 0)


# ---------- Setup ----------
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
aruco_params.adaptiveThreshWinSizeStep = 14
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

print("Starting camera...")
cap = CameraThread(VIDEO_ID)
print("Waiting for first frame...")
first_frame = None
while first_frame is None:
    first_frame = cap.read()
print("Ready.")

WINDOW_HEIGHT, WINDOW_WIDTH = first_frame.shape[:2]
SCALE = min(WINDOW_WIDTH, WINDOW_HEIGHT) / 720.0
_overlay_buf = np.empty_like(first_frame)

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, caption="AR Game")
# Create a texture to hold the display image, which will be updated each frame
_display_tex = pyglet.image.Texture.create(WINDOW_WIDTH, WINDOW_HEIGHT)
display_sprite = pyglet.sprite.Sprite(_display_tex)

state = {
    'display': first_frame,
    'last_board': None,
    'stale_frames': 0,
    'board_ever_detected': False,
}

game = WhackAMoleGame()

pyglet.clock.schedule_interval(update, 1 / UPDATE_HZ)


@window.event
def on_draw():
    window.clear()
    display_sprite.draw()


@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.Q:
        window.close()
    elif symbol == pyglet.window.key.R:
        game.start_round()


@window.event
def on_close():
    cap.release()


pyglet.app.run()