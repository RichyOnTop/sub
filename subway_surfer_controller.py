#!/usr/bin/env python3
"""
Subway Surfers Webcam Gesture Controller
========================================
Controls Subway Surfers on Poki using body gestures via webcam.
Uses MediaPipe PoseLandmarker (Tasks API) and pynput / ydotool.

Fixes in this version:
  - EMA smoothing on landmarks (stops jitter)
  - Velocity-based jump detection (rises quickly = jump)
  - Calibration phase with on-screen progress bar
  - Press 'c' to re-calibrate baseline at any time
  - On-screen debug numbers (top-right corner)
  - draw_landmarks uses SMOOTHED data (was using raw before)

Install:
    pip install opencv-python mediapipe pynput numpy

First run auto-downloads pose_landmarker_lite.task (~2 MB).
"""

import cv2
import time
import enum
import subprocess
import json
import os
import urllib.request
import numpy as np
import argparse
import sys

# ----------------------------------------------------------------------- #
# Backend detection                                                      #
# ----------------------------------------------------------------------- #
try:
    from pynput.keyboard import Key, Controller as KeyboardController
    PYNPUT_OK = True
except ImportError:
    PYNPUT_OK = False

YDO_TOOL_OK = False
try:
    subprocess.run(["which", "ydotool"], capture_output=True, check=True)
    YDO_TOOL_OK = True
except (subprocess.CalledProcessError, FileNotFoundError):
    YDO_TOOL_OK = False

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False


# ======================================================================= #
# Config                                                                #
# ======================================================================= #
class Config:
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

    MODEL_FILENAME = "pose_landmarker_lite.task"
    MODEL_URL = (
        "https://storage.googleapis.com/"
        "mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    )

    # Smoothing: exponential moving average alpha
    # 0.2 = very smooth, slight lag.  0.5 = responsive, some jitter.
    EMA_ALPHA = 0.25

    # Calibration: stand still for this many seconds at start (or when 'c' pressed)
    CALIB_SECONDS = 5.0

    # --- Detection thresholds ---
    # Jump: head velocity upward (normalised units per second)
    # Typical jump: velocity ~0.3 to 1.0.  Set low to be sensitive.
    JUMP_VELOCITY_THRESHOLD = 0.12

    # Jump: also trigger if head is this far above baseline (fallback)
    JUMP_POSITION_THRESHOLD = 0.05

    # Duck: head and hips this far below baseline
    DUCK_POSITION_THRESHOLD = 0.05

    # Lean: shoulder centre this far from frame centre
    LEAN_THRESHOLD = 0.07

    # Timing
    KEY_TAP_DURATION = 0.10
    ACTION_COOLDOWN = 0.30
    STABILITY_FRAMES = 4    # consecutive frames with same action before triggering

    # Visual
    WINDOW_NAME = "Subway Surfers Controller"
    BAR_HEIGHT = 40

    BROWSER_NAMES = ["brave", "chromium", "chrome", "firefox", "google-chrome"]


# ======================================================================= #
# Action enum                                                          #
# ======================================================================= #
class Action(enum.Enum):
    NONE = "Idle"
    JUMP = "Jump"
    DUCK = "Duck"
    LEFT = "Left"
    RIGHT = "Right"


# ======================================================================= #
# PoseDetector — MediaPipe Tasks API (IMAGE mode, synchronous)          #
# ======================================================================= #
class PoseDetector:
    def __init__(self, model_path=None):
        if not MEDIAPIPE_OK:
            raise RuntimeError("MediaPipe not installed. pip install mediapipe")

        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, Config.MODEL_FILENAME)

        if not os.path.exists(model_path):
            print("[PoseDetector] Downloading model to", model_path)
            try:
                urllib.request.urlretrieve(Config.MODEL_URL, model_path)
                print("[PoseDetector] Download complete.")
            except Exception as e:
                print("[PoseDetector] FAILED to download:", e)
                print("  Download manually and place at:", model_path)
                raise

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            output_segmentation_masks=False,
        )

        self.landmarker = PoseLandmarker.create_from_options(options)
        print("[PoseDetector] PoseLandmarker initialised (IMAGE mode).")

        # Cache drawing helpers (new API location)
        self._du = mp.tasks.vision.drawing_utils
        self._style = mp.tasks.vision.drawing_styles
        self._connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

    def detect(self, frame_bgr):
        """Synchronous pose detection. Returns PoseLandmarkerResult or None."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        try:
            return self.landmarker.detect(mp_image)
        except Exception as e:
            print("[PoseDetector] detect() error:", e)
            return None

    def draw_landmarks(self, frame_bgr, smoothed_landmarks):
        """
        Draw pose skeleton onto frame_bgr using OpenCV primitives.
        `smoothed_landmarks` is a list of (x,y,z) tuples (normalised 0..1).
        Returns annotated BGR frame.
        """
        if smoothed_landmarks is None:
            return frame_bgr

        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]

        # Draw landmarks as green circles
        for (x, y, z) in smoothed_landmarks:
            cx = int(x * w)
            cy = int(y * h)
            cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)

        # Draw pose connections as green lines
        POSE_CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,7),
            (0,4),(4,5),(5,6),(6,8),
            (9,10),
            (11,12),(11,13),(13,15),(15,17),
            (15,19),(15,21),(17,19),
            (12,14),(14,16),(16,18),
            (16,20),(16,22),(18,20),
            (11,23),(12,24),(23,24),
            (23,25),(24,26),(25,27),
            (26,28),(27,29),(28,30),
            (29,31),(30,32),
        ]
        for (a, b) in POSE_CONNECTIONS:
            if a >= len(smoothed_landmarks) or b >= len(smoothed_landmarks):
                continue
            x1 = int(smoothed_landmarks[a][0] * w)
            y1 = int(smoothed_landmarks[a][1] * h)
            x2 = int(smoothed_landmarks[b][0] * w)
            y2 = int(smoothed_landmarks[b][1] * h)
            cv2.line(annotated, (x1,y1), (x2,y2), (0,255,0), 2)

        # Draw baseline as red horizontal line
        if smoothed_landmarks and hasattr(self, "_debug_baseline_y"):
            by = int(self._debug_baseline_y * h)
            cv2.line(annotated, (0,by), (w,by), (0,0,255), 1)

        return annotated

    def close(self):
        if hasattr(self, "landmarker"):
            self.landmarker.close()


# ======================================================================= #
# SmoothedLandmarks — exponential moving average                     #
# ======================================================================= #
class SmoothedLandmarks:
    """
    Applies an exponential moving average to pose landmarks to remove jitter.
    Also computes head velocity (dy/dt) for jump detection.
    """
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self._smoothed = None       # list of (x, y, z) tuples
        self._prev_head_y = None
        self._velocity = 0.0        # head Y velocity (normalised units / second)
        self._prev_time = None
        self._ready = False

    def update(self, pose_landmarks_list):
        """
        Update with a new detection.
        `pose_landmarks_list` is result.pose_landmarks
        (list of lists of NormalizedLandmark objects).
        Returns the smoothed landmark list as [(x,y,z), ...].
        """
        now = time.time()
        raw = [(lm.x, lm.y, lm.z) for lm in pose_landmarks_list[0]]

        if self._smoothed is None:
            self._smoothed = list(raw)
        else:
            a = self.alpha
            for i, (x, y, z) in enumerate(raw):
                osx, osy, osz = self._smoothed[i]
                self._smoothed[i] = (
                    osx + a * (x - osx),
                    osy + a * (y - osy),
                    osz + a * (z - osz),
                )

        # Compute velocity: head_y DECREASES when you raise your head
        # (Y=0 is top of frame, Y=1 is bottom)
        # So head rising → head_y gets SMALLER → velocity = prev_y - current_y (positive = rising)
        if self._prev_head_y is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 0.005:
                self._velocity = (self._prev_head_y - self._smoothed[0][1]) / dt

        self._prev_head_y = self._smoothed[0][1]
        self._prev_time = now
        self._ready = True

        return self._smoothed

    @property
    def head_y(self):
        return self._smoothed[0][1] if self._smoothed else 0.5

    @property
    def shoulder_cx(self):
        if not self._smoothed or len(self._smoothed) < 25:
            return 0.5
        return (self._smoothed[11][0] + self._smoothed[12][0]) / 2.0

    @property
    def hip_cy(self):
        if not self._smoothed or len(self._smoothed) < 25:
            return 0.5
        return (self._smoothed[23][1] + self._smoothed[24][1]) / 2.0

    @property
    def velocity_up(self):
        """Positive = head rising (normalised units / second)."""
        return self._velocity

    @property
    def ready(self):
        return self._ready


# ======================================================================= #
# GestureInterpreter                                                    #
# ======================================================================= #
class GestureInterpreter:
    """
    Interprets smoothed pose data into game actions.
    """
    def __init__(self, config):
        self.cfg = config
        self._smoother = SmoothedLandmarks(alpha=config.EMA_ALPHA)

        # Baseline (set during calibration)
        self.baseline_y = None
        self._calib_buffer = []
        self._calib_start = None   # set in Application.run()
        self._calib_active = False

        self._pending_action = Action.NONE
        self._action_counter = 0
        self._last_action_time = 0.0

    def start_calibration(self):
        """Call this to (re-)start calibration (e.g. when user presses 'c')."""
        self.baseline_y = None
        self._calib_buffer = []
        self._calib_start = time.time()
        self._calib_active = True
        print("[Calib] Re-calibration started — stand still for {:.0f}s.".format(
            Config.CALIB_SECONDS))

    def _update_baseline(self, head_y):
        """During calibration, collect samples. Lock baseline when done."""
        if self.baseline_y is not None:
            return   # already calibrated
        if self._calib_start is None:
            self._calib_start = time.time()
            self._calib_active = True
            return
        elapsed = time.time() - self._calib_start
        if elapsed < Config.CALIB_SECONDS:
            self._calib_buffer.append(head_y)
        elif len(self._calib_buffer) > 0:
            self.baseline_y = float(np.mean(self._calib_buffer))
            self._calib_active = False
            print("[Calib] Baseline head Y = {:.4f}".format(self.baseline_y))

    def interpret(self, result, timestamp):
        """Read PoseLandmarker result and return the current Action."""

        # --- No pose detected ---
        if result is None or not result.pose_landmarks:
            self._pending_action = Action.NONE
            self._action_counter = 0
            return Action.NONE

        # --- Smooth landmarks ---
        smoothed = self._smoother.update(result.pose_landmarks)

        head_y = self._smoother.head_y
        shoulder_cx = self._smoother.shoulder_cx
        hip_cy = self._smoother.hip_cy
        vel_up = self._smoother.velocity_up

        # --- Calibration ---
        if self.baseline_y is None:
            self._update_baseline(head_y)
            return Action.NONE

        # --- Decide action ---
        action = Action.NONE
        dy = self.baseline_y - head_y   # positive = head above baseline

        # Jump: velocity UP (most reliable) OR absolute position above baseline
        if vel_up > self.cfg.JUMP_VELOCITY_THRESHOLD:
            action = Action.JUMP
        elif dy > self.cfg.JUMP_POSITION_THRESHOLD:
            action = Action.JUMP
        # Duck: head AND hips below baseline
        elif (head_y - self.baseline_y > self.cfg.DUCK_POSITION_THRESHOLD and
              hip_cy - self.baseline_y > self.cfg.DUCK_POSITION_THRESHOLD * 0.5):
            action = Action.DUCK
        # Lean left / right
        elif shoulder_cx < 0.5 - self.cfg.LEAN_THRESHOLD:
            action = Action.LEFT
        elif shoulder_cx > 0.5 + self.cfg.LEAN_THRESHOLD:
            action = Action.RIGHT

        # --- Stability filter ---
        if action == self._pending_action:
            self._action_counter += 1
        else:
            self._pending_action = action
            self._action_counter = 1

        if self._action_counter < self.cfg.STABILITY_FRAMES:
            return Action.NONE

        # --- Cooldown ---
        if (action != Action.NONE and
                timestamp - self._last_action_time < self.cfg.ACTION_COOLDOWN):
            return Action.NONE

        if action != Action.NONE:
            self._last_action_time = timestamp
            if action == Action.JUMP:
                print("[Action] JUMP  (vel={:.3f}, dy={:.3f})".format(
                    vel_up, dy))
            else:
                print("[Action]", action.name)

        return action

    def calibration_progress(self):
        """Return (elapsed_seconds, total_seconds, done_bool)."""
        if self.baseline_y is not None:
            return Config.CALIB_SECONDS, Config.CALIB_SECONDS, True
        if self._calib_start is None:
            return 0.0, Config.CALIB_SECONDS, False
        elapsed = time.time() - self._calib_start
        return elapsed, Config.CALIB_SECONDS, False


# ======================================================================= #
# KeySender — pynput (primary) -> ydotool (fallback)               #
# ======================================================================= #
class KeySender:
    def __init__(self, config):
        self.cfg = config
        self._backend = "none"
        self._kb = None
        self._focus_cache = False
        self._cache_until = 0.0

        if PYNPUT_OK:
            try:
                self._kb = KeyboardController()
                self._backend = "pynput"
                print("[KeySender] Using pynput backend.")
            except Exception as e:
                print("[KeySender] pynput init failed:", e)

        if self._backend == "none" and YDO_TOOL_OK:
            try:
                subprocess.run(["ydotool", "key", "0"],
                              capture_output=True, timeout=2)
                self._backend = "ydotool"
                print("[KeySender] Using ydotool backend.")
            except Exception as e:
                print("[KeySender] ydotool test failed:", e)

        if self._backend == "none":
            print(
                "[KeySender] WARNING: No key-backend!\n"
                "  pip install pynput\n"
                "  OR: sudo pacman -S ydotool && sudo systemctl enable --now ydotool"
            )

    def is_browser_focused(self):
        now = time.time()
        if now < self._cache_until:
            return self._focus_cache
        focused = False
        try:
            out = subprocess.run(
                ["hyprctl", "activewindow", "-j"],
                capture_output=True, text=True, timeout=1
            ).stdout.strip()
            if out:
                info = json.loads(out)
                cls = info.get("class", "").lower()
                for name in self.cfg.BROWSER_NAMES:
                    if name in cls:
                        focused = True
                        break
        except Exception:
            focused = True

        self._focus_cache = focused
        self._cache_until = now + 0.5
        return focused

    def send_action(self, action):
        if action == Action.NONE:
            return False
        if not self.is_browser_focused():
            return False
        if self._backend == "none":
            return False

        arrow = {Action.JUMP: "up", Action.DUCK: "down",
                   Action.LEFT: "left", Action.RIGHT: "right"}.get(action)
        if arrow is None:
            return False

        try:
            if self._backend == "pynput":
                return self._send_pynput(arrow)
            elif self._backend == "ydotool":
                return self._send_ydotool(arrow)
        except Exception as e:
            print("[KeySender] Error sending '{}':".format(arrow), e)
        return False

    def _send_pynput(self, arrow):
        k = getattr(Key, arrow)
        self._kb.press(k)
        time.sleep(self.cfg.KEY_TAP_DURATION)
        self._kb.release(k)
        return True

    def _send_ydotool(self, arrow):
        code_map = {"up": "103", "down": "108", "left": "105", "right": "106"}
        code = code_map[arrow]
        subprocess.run(
            ["ydotool", "key", "{}:1".format(code), "{}:0".format(code)],
            check=True, capture_output=True, timeout=2
        )
        return True


# ======================================================================= #
# HUD                                                                  #
# ======================================================================= #
class HUD:
    def __init__(self, config):
        self.cfg = config
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(self, frame, focus, action, fps, backend, interpreter):
        """Draw full HUD overlay. `interpreter` is used to show debug numbers."""
        h, w = frame.shape[:2]
        bar_h = self.cfg.BAR_HEIGHT

        # --- Top status bar ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        focus_str = "Browser" if focus else "N/A (click browser!)"
        text = (
            "Focus: {}  |  Action: {}  |  "
            "Backend: {}  |  FPS: {:.1f}  |  'q' quit  |  'c' calibrate"
            .format(focus_str, action.value, backend, fps)
        )
        cv2.putText(frame, text, (10, bar_h - 10),
                     self._font, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # --- Calibration progress bar ---
        elapsed, total, done = interpreter.calibration_progress()
        if not done:
            prog = min(elapsed / total, 1.0)
            bar_w = int(w * 0.6)
            bar_x = int(w * 0.2)
            bar_y = h // 2 - 20
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + bar_w, bar_y + 24),
                         (60, 60, 60), -1)
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + int(bar_w * prog), bar_y + 24),
                         (0, 180, 255), -1)
            cv2.putText(frame,
                         "CALIBRATING - Stand still! {:.0f}s / {:.0f}s".format(
                             min(elapsed, total), total),
                         (bar_x, bar_y - 8),
                         self._font, 0.5, (0, 180, 255), 2, cv2.LINE_AA)

        # --- On-screen debug numbers (top-right) ---
        if done and interpreter._smoother.ready:
            sm = interpreter._smoother
            by = interpreter.baseline_y or 0.0
            dy = by - sm.head_y
            dbg_lines = [
                "head_y: {:.3f}".format(sm.head_y),
                "baseline: {:.3f}".format(by),
                "dy: {:.3f}".format(dy),
                "vel_up: {:.3f}".format(sm.velocity_up),
                "shoulder_cx: {:.3f}".format(sm.shoulder_cx),
            ]
            x0 = w - 220
            for i, line in enumerate(dbg_lines):
                cv2.putText(frame, line, (x0, 60 + i * 18),
                             self._font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        # --- Gesture hints (bottom-left) ---
        hints = [
            "Jump: raise hands above shoulders (or jump up!)",
            "Duck: crouch / bend knees",
            "Left: lean or step left",
            "Right: lean or step right",
        ]
        y0 = h - 12 * len(hints) - 8
        for i, hint in enumerate(hints):
            cv2.putText(frame, hint, (10, y0 + i * 14),
                        self._font, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        return frame


# ======================================================================= #
# Application                                                          #
# ======================================================================= #
class Application:
    def __init__(self):
        self.cfg = Config()
        self.detector = PoseDetector()
        self.interpreter = GestureInterpreter(self.cfg)
        self.sender = KeySender(self.cfg)
        self.hud = HUD(self.cfg)
        self.cap = None
        self._fps_buffer = []

    def _open_camera(self):
        self.cap = cv2.VideoCapture(self.cfg.CAMERA_INDEX)
        if not self.cap.isOpened():
            print("[Camera] Could not open camera", self.cfg.CAMERA_INDEX)
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.CAMERA_HEIGHT)
        print("[Camera] Opened camera", self.cfg.CAMERA_INDEX)
        return True

    def _compute_fps(self):
        now = time.time()
        self._fps_buffer.append(now)
        if len(self._fps_buffer) > 30:
            self._fps_buffer.pop(0)
        if len(self._fps_buffer) < 2:
            return 0.0
        return len(self._fps_buffer) / (self._fps_buffer[-1] - self._fps_buffer[0])

    def run(self):
        print("=" * 62)
        print("  Subway Surfers - Webcam Gesture Controller")
        print("=" * 62)
        print()
        print("Before starting:")
        print("  1. Open Brave to the Subway Surfers Poki page.")
        print("  2. Start the game (past menus to the track).")
        print("  3. Stand ~1 m from webcam, full body visible.")
        print("  4. When the window opens, STAND STILL for 5 seconds.")
        print("     A blue progress bar will show calibration progress.")
        print("  5. Press 'c' at any time to re-calibrate.")
        print()
        print("Press 'q' in the webcam window to quit.")
        print("-" * 62)

        if not MEDIAPIPE_OK:
            print("\n[FATAL] MediaPipe not found. pip install mediapipe\n")
            return

        if self.sender._backend == "none":
            print("\n[WARNING] No key-sending backend! Keys will NOT be sent.\n")

        if not self._open_camera():
            return

        # Start calibration timer
        self.interpreter.start_calibration()

        cv2.namedWindow(self.cfg.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.cfg.WINDOW_NAME,
                         self.cfg.CAMERA_WIDTH,
                         self.cfg.CAMERA_HEIGHT + self.cfg.BAR_HEIGHT)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame = cv2.flip(frame, 1)
                timestamp = time.time()

                # --- Pose detection ---
                result = self.detector.detect(frame)

                # --- Smooth landmarks (for gesture interpreter AND drawing) ---
                if result is not None and result.pose_landmarks:
                    smoothed = self.interpreter._smoother.update(result.pose_landmarks)
                else:
                    smoothed = None

                # --- Draw skeleton (using SMOOTHED landmarks) ---
                if smoothed is not None:
                    # Pass smoothed data to detector's draw function
                    self.detector._debug_baseline_y = self.interpreter.baseline_y
                    frame = self.detector.draw_landmarks(frame, smoothed)

                # --- Interpret gesture ---
                action = self.interpreter.interpret(result, timestamp)

                # --- Send key ---
                sent = False
                if action != Action.NONE and self.interpreter.baseline_y is not None:
                    sent = self.sender.send_action(action)

                # --- HUD ---
                fps = self._compute_fps()
                focus = self.sender.is_browser_focused()
                display_action = action if (sent and self.interpreter.baseline_y is not None) else Action.NONE
                if self.interpreter.baseline_y is None:
                    display_action = Action.NONE
                frame = self.hud.draw(
                    frame, focus, display_action, fps,
                    self.sender._backend, self.interpreter
                )

                cv2.imshow(self.cfg.WINDOW_NAME, frame)

                # --- Key handling ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("Quitting...")
                    break
                elif key == ord('c') or key == ord('C'):
                    self.interpreter.start_calibration()

        finally:
            if self.cap is not None:
                self.cap.release()
            self.detector.close()
            cv2.destroyAllWindows()
            print("Bye!")


# ======================================================================= #
# Entry point                                                          #
# ======================================================================= #
def test_cameras():
    """Try opening camera indices 0-9 and print which ones work."""
    print("Testing camera indices 0-9...")
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("  [OK] Camera index {} - {}x{}".format(i, w, h))
            cap.release()
        else:
            print("  [NO] Camera index {} - cannot open".format(i))
    print()
    print("Tip: If your second camera shows as index 2, run:")
    print("     python subway_surfer_controller.py --camera 2")


def main():
    parser = argparse.ArgumentParser(
        description="Subway Surfers Webcam Gesture Controller"
    )
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Camera index to use (overrides Config.CAMERA_INDEX)"
    )
    parser.add_argument(
        "--test-cameras", action="store_true",
        help="List available camera indices and exit"
    )
    args = parser.parse_args()

    if args.test_cameras:
        test_cameras()
        sys.exit(0)

    if args.camera is not None:
        Config.CAMERA_INDEX = args.camera
        print("[Config] Camera index overridden to", Config.CAMERA_INDEX)

    app = Application()
    app.run()


if __name__ == "__main__":
    main()
