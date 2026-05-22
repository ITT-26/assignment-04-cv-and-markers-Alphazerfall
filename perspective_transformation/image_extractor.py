import argparse
import sys
import cv2
import numpy as np

SELECTION_WINDOW_TITLE = "Select 4 corner points  (ESC: reset)"
RESULT_WINDOW_TITLE = "Warped result  (S: save, ESC: redo)"


class ImageExtractor:
    def __init__(self, input_path, output_path, width=None, height=None):
        self.input_path, self.output_path = input_path, output_path
        self.forced_width, self.forced_height = width, height
        self.source_image = cv2.imread(input_path)
        if self.source_image is None:
            raise FileNotFoundError(f"Could not load image: {input_path}")

        self.points = []
        self.warped = None

    def _draw_overlay(self):
        display = self.source_image.copy()

        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                cv2.line(display, self.points[i], self.points[i + 1], (0, 255, 0), 2, cv2.LINE_AA)
            if len(self.points) == 4:
                # close the polygon by connecting the last point to the first
                cv2.line(display, self.points[3], self.points[0], (0, 255, 0), 2, cv2.LINE_AA)

        for i, p in enumerate(self.points):
            cv2.circle(display, p, 6, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(display, p, 7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display, str(i + 1), (p[0] + 10, p[1] - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1,
                        cv2.LINE_AA)

        # status banner in the top-left corner
        msg = f"Points: {len(self.points)}/4   (ESC: reset, Q: quit)"
        cv2.putText(display, msg, (10, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, msg, (10, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)

        cv2.imshow(SELECTION_WINDOW_TITLE, display)

    def _warp_and_show(self):
        src = self._order_points(self.points)
        tl, tr, br, bl = src

        if self.forced_width is not None and self.forced_height is not None:
            # User passed --width/--height: use them verbatim.
            width, height = self.forced_width, self.forced_height
        else:
            # Pick the longer of the opposite edges so the output doesn't
            # lose resolution on whichever side is closer to the camera.
            width_top    = np.linalg.norm(tr - tl)
            width_bottom = np.linalg.norm(br - bl)
            height_left  = np.linalg.norm(bl - tl)
            height_right = np.linalg.norm(br - tr)

            width  = int(round(max(width_top, width_bottom)))
            height = int(round(max(height_left, height_right)))

        dst = np.array([
            [0,         0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0,         height - 1],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        self.warped = cv2.warpPerspective(self.source_image, M, (width, height))
        cv2.imshow(RESULT_WINDOW_TITLE, self.warped)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            if len(self.points) == 4:
                self._warp_and_show()

    def _order_points(self, pts):
        pts = np.array(pts, dtype="float32")
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left  (smallest x+y)
        rect[2] = pts[np.argmax(s)]   # bottom-right (largest x+y)
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right (smallest y-x)
        rect[3] = pts[np.argmax(diff)]  # bottom-left (largest y-x)
        return rect

    def _reset(self):
        self.points.clear()
        self.warped = None
        if cv2.getWindowProperty(RESULT_WINDOW_TITLE, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow(RESULT_WINDOW_TITLE)

    def run(self):
        cv2.namedWindow(SELECTION_WINDOW_TITLE)
        cv2.setMouseCallback(SELECTION_WINDOW_TITLE, self._mouse_callback)
        print("Click 4 corners of the region you want to extract.")
        print("  ESC: reset selection   S: save warped image   Q: quit")

        while True:
            # Quit if the selection window was closed via the X button
            if cv2.getWindowProperty(SELECTION_WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break

            self._draw_overlay()

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to reset
                self._reset()
            elif key == ord('s') and self.warped is not None:  # S to save
                cv2.imwrite(self.output_path, self.warped)
                print(f"Saved warped image to: {self.output_path}")
                break
            elif key == ord('q'):  # Q to quit without saving
                break

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Extract a perspective-warped image from an image.")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("output", help="Path to save the warped output image")
    parser.add_argument("--width", type=int, default=None,
                        help="Output width in pixels (default: auto from selected points)")
    parser.add_argument("--height", type=int, default=None,
                        help="Output height in pixels (default: auto from selected points)")
    args = parser.parse_args()

    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be specified together.")

    try:
        extractor = ImageExtractor(args.input, args.output, args.width, args.height)
        extractor.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()