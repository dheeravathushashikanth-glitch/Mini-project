"""
Unified hand-gesture desktop control — single application.

Uses MediaPipe HandLandmarker (Tasks API), OpenCV, and PyAutoGUI. Combines:
  • Distance-based shortcuts (volume, browser tabs, swipe navigation)
  • Pose-based control (fist drag, V-gesture cursor, two-finger scroll, pinch scroll / h-scroll)
  • Role-based pointer: Right hand moves cursor; Left hand click / double / right-click

MediaPipe 0.10.30+ exposes only the Tasks API (no ``mp.solutions``).
"""

from __future__ import annotations

import math
import sys
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pyautogui

from mediapipe.tasks.python.core import base_options as mp_base_options
from mediapipe.tasks.python.vision import hand_landmarker as mp_hand_landmarker
from mediapipe.tasks.python.vision.core.image import Image as MpImage
from mediapipe.tasks.python.vision.core.image import ImageFormat
from mediapipe.tasks.python.vision.core import vision_task_running_mode as mp_vision_mode

# -----------------------------------------------------------------------------
# Model (downloaded once under repo /models)
# -----------------------------------------------------------------------------
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILENAME = "hand_landmarker.task"

# -----------------------------------------------------------------------------
# Landmarks (MediaPipe hand topology)
# -----------------------------------------------------------------------------
LM_WRIST = 0
LM_THUMB_TIP = 4
LM_THUMB_IP = 3
LM_INDEX_TIP = 8
LM_INDEX_PIP = 6
LM_INDEX_MCP = 5
LM_MIDDLE_TIP = 12
LM_MIDDLE_PIP = 10
LM_MIDDLE_MCP = 9
LM_RING_TIP = 16
LM_RING_PIP = 14
LM_PINKY_TIP = 20
LM_PINKY_PIP = 18

@dataclass(frozen=True)
class AppConfig:
    """Centralized configuration for the Simple Gesture Set."""
    # Detection
    PINCH_MAX_DIST: int = 35       # 👌 Pinch (Thumb + Index)
    VSIGN_MIN_DIST: int = 60       # ✌️ V-sign (Index + Middle)
    PALM_FINGERS_MIN: int = 4      # ✋ Open Palm
    
    # Motion / Hold
    MOTION_DELTA_MIN: int = 12     # Pixels moved to trigger Scroll/Volume
    SCROLL_SENSITIVITY: int = 3    # Scroll amount per movement
    
    # Clicks
    TAP_Y_SPIKE: int = 15          # Sudden Index movement for Left Click
    RIGHT_CLICK_HOLD_S: float = 1.0 # Duration to hold pinch for Right Click
    
    # UI
    HISTORY_SIZE: int = 5
    WINDOW_TITLE_PERSISTENCE: float = 2.0

CONFIG = AppConfig()

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _models_directory() -> Path:
    return _repo_root() / "models"


def ensure_hand_landmarker_model() -> Path:
    models_dir = _models_directory()
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / MODEL_FILENAME
    if model_path.is_file() and model_path.stat().st_size > 0:
        return model_path
    print(f"Downloading {MODEL_FILENAME} (first run only)...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print(f"Saved to {model_path}")
    return model_path


def lm_to_px(lm, w: int, h: int) -> tuple[int, int]:
    return int(lm.x * w), int(lm.y * h)


def dist_norm(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def dist_px(p: tuple[int, int], q: tuple[int, int]) -> float:
    return math.hypot(q[0] - p[0], q[1] - p[1])


def finger_extended_from_wrist(lm, tip_i: int, pip_i: int, wrist_i: int = LM_WRIST) -> bool:
    d_tip = dist_norm(lm[tip_i], lm[wrist_i])
    d_pip = dist_norm(lm[pip_i], lm[wrist_i])
    return d_tip > d_pip * 1.18


def thumb_extended(lm) -> bool:
    d_tip = dist_norm(lm[LM_THUMB_TIP], lm[LM_WRIST])
    d_ip = dist_norm(lm[LM_THUMB_IP], lm[LM_WRIST])
    return d_tip > d_ip * 1.12


def finger_states(lm) -> dict[str, bool]:
    return {
        "thumb": thumb_extended(lm),
        "index": finger_extended_from_wrist(lm, LM_INDEX_TIP, LM_INDEX_PIP),
        "middle": finger_extended_from_wrist(lm, LM_MIDDLE_TIP, LM_MIDDLE_PIP),
        "ring": finger_extended_from_wrist(lm, LM_RING_TIP, LM_RING_PIP),
        "pinky": finger_extended_from_wrist(lm, LM_PINKY_TIP, LM_PINKY_PIP),
    }


def classify_hand_pose(fs: dict[str, bool]) -> GestureId:
    """Map extended fingers to a single dominant pose label."""
    n_up = sum(1 for k in ("index", "middle", "ring", "pinky") if fs[k])
    if n_up == 0 and not fs["thumb"]:
        return GestureId.FIST
    if n_up == 1:
        if fs["pinky"]:
            return GestureId.PINKY
        if fs["ring"]:
            return GestureId.RING
        if fs["middle"]:
            return GestureId.MID
        if fs["index"]:
            return GestureId.INDEX
    if fs["index"] and fs["middle"] and not fs["ring"] and not fs["pinky"]:
        return GestureId.V_GEST
    return GestureId.NONE


@dataclass
class HandSnapshot:
    label: str
    lm: list
    px: list[tuple[int, int]]
    fs: dict[str, bool]
    d_index_middle: float
    d_thumb_index: float
    gesture: int # count of open fingers


def build_hand_snapshot(label: str, lm, w: int, h: int) -> HandSnapshot:
    px = [lm_to_px(lm[i], w, h) for i in range(21)]
    fs = finger_states(lm)
    d_im = dist_px(px[LM_INDEX_TIP], px[LM_MIDDLE_TIP])
    d_ti = dist_px(px[LM_THUMB_TIP], px[LM_INDEX_TIP])
    n_up = sum(1 for k in fs if fs[k])
    return HandSnapshot(
        label=label,
        lm=lm,
        px=px,
        fs=fs,
        d_index_middle=d_im,
        d_thumb_index=d_ti,
        gesture=n_up,
    )


@dataclass
class ControlState:
    """Simplified state machine for 7-operation set."""
    prev_hand_y: dict[str, float] = field(default_factory=dict)
    pinch_start_time: dict[str, float] = field(default_factory=dict)
    index_prev_y: dict[str, float] = field(default_factory=dict)
    last_click_time: float = 0.0
    
    history: deque[str] = field(default_factory=lambda: deque(maxlen=CONFIG.HISTORY_SIZE))
    last_event_time: float = 0.0

    def add_event(self, event: str, now: float) -> None:
        if not self.history or self.history[-1] != event:
            self.history.append(event)
        self.last_event_time = now

    def get_title_status(self, now: float) -> str:
        if not self.history: return ""
        return " | ".join(list(self.history))


def _held_reset(state: ControlState, key: str) -> None:
    state.held_actions[key] = False


def pulse_action(
    state: ControlState,
    key: str,
    condition: bool,
    now: float,
    min_interval: float,
    action: Callable[[], None],
) -> bool:
    if not condition:
        _held_reset(state, key)
        return False
    if state.held_actions.get(key):
        return False
    if now - state.last_fire_time.get(key, 0.0) < min_interval:
        return False
    state.held_actions[key] = True
    state.last_fire_time[key] = now
    action()
    return True


def draw_green_pair(frame: np.ndarray, p: tuple[int, int], q: tuple[int, int]) -> None:
    """Green segment between fingertips used for distance-based rules."""
    cv2.line(frame, p, q, (0, 255, 0), 2)


def draw_hand_connections_bgr(
    frame_bgr: np.ndarray,
    normalized_landmarks: list,
    connections: list,
    line_color: tuple[int, int, int] = (0, 200, 255),
    point_color: tuple[int, int, int] = (255, 0, 0),
) -> None:
    h, w = frame_bgr.shape[:2]
    pts: list[tuple[int, int]] = []
    for lm in normalized_landmarks:
        pts.append((int(lm.x * w), int(lm.y * h)))
    for conn in connections:
        cv2.line(frame_bgr, pts[conn.start], pts[conn.end], line_color, 2)
    for p in pts:
        cv2.circle(frame_bgr, p, 3, point_color, -1)


def detect_gestures(
    frame: np.ndarray,
    hands: list[HandSnapshot],
    state: ControlState,
    now: float,
    screen_w: int,
    screen_h: int,
) -> list[str]:
    """Simple 7-gesture set implementation."""
    hud: list[str] = []
    
    for hs in hands:
        label = hs.label
        px = hs.px
        n_up = hs.gesture # count_open_fingers from snapshot
        
        # 1. Cursor Move: Open Palm (✋)
        if n_up >= CONFIG.PALM_FINGERS_MIN:
            t8 = hs.lm[LM_INDEX_TIP]
            sx, sy = int(t8.x * screen_w), int(t8.y * screen_h)
            pyautogui.moveTo(sx, sy)
            draw_neon_glow(frame, px[LM_INDEX_TIP], (255, 200, 0), 10)
            continue # Prioritize Move
            
        # 2. Volume: V-Sign (✌️)
        dist_im = hs.d_index_middle
        if hs.fs["index"] and hs.fs["middle"] and not hs.fs["ring"] and dist_im < CONFIG.VSIGN_MIN_DIST:
            draw_green_pair(frame, px[LM_INDEX_TIP], px[LM_MIDDLE_TIP])
            curr_y = px[LM_MIDDLE_TIP][1]
            prev_y = state.prev_hand_y.get(label)
            if prev_y is not None:
                dy = curr_y - prev_y
                if dy < -CONFIG.MOTION_DELTA_MIN:
                    pyautogui.press("volumeup")
                    state.add_event(f"{label} VOL UP", now)
                    hud.append("VOL UP")
                elif dy > CONFIG.MOTION_DELTA_MIN:
                    pyautogui.press("volumedown")
                    state.add_event(f"{label} VOL DOWN", now)
                    hud.append("VOL DOWN")
            state.prev_hand_y[label] = curr_y
            continue

        # 3. Pinch: Scroll/Right-Click (👌)
        dist_ti = hs.d_thumb_index
        if dist_ti < CONFIG.PINCH_MAX_DIST:
            draw_green_pair(frame, px[LM_THUMB_TIP], px[LM_INDEX_TIP])
            
            # Check for Motion first (Scrolling)
            curr_y = px[LM_INDEX_TIP][1]
            prev_y = state.prev_hand_y.get(label)
            moved = False
            if prev_y is not None:
                dy = curr_y - prev_y
                if dy < -CONFIG.MOTION_DELTA_MIN:
                    pyautogui.scroll(CONFIG.SCROLL_SENSITIVITY)
                    state.add_event("Scroll UP", now)
                    hud.append("SCROLL UP")
                    moved = True
                elif dy > CONFIG.MOTION_DELTA_MIN:
                    pyautogui.scroll(-CONFIG.SCROLL_SENSITIVITY)
                    state.add_event("Scroll DOWN", now)
                    hud.append("SCROLL DOWN")
                    moved = True
            
            state.prev_hand_y[label] = curr_y
            
            # Check for Hold (Right Click) if not moving
            if not moved:
                if label not in state.pinch_start_time:
                    state.pinch_start_time[label] = now
                elif now - state.pinch_start_time[label] > CONFIG.RIGHT_CLICK_HOLD_S:
                    pyautogui.rightClick()
                    state.add_event("Right Click", now)
                    hud.append("RIGHT CLICK")
                    state.pinch_start_time[label] = now + 999 # Cooldown
            else:
                state.pinch_start_time.pop(label, None)
            continue
        else:
            state.pinch_start_time.pop(label, None)

        # 4. Left Click: Index Tap (👆)
        if hs.fs["index"] and n_up == 1:
            curr_y = px[LM_INDEX_TIP][1]
            prev_y = state.index_prev_y.get(label)
            if prev_y is not None:
                dy = curr_y - prev_y
                # Detect sudden downward motion (tapping)
                if dy > CONFIG.TAP_Y_SPIKE and (now - state.last_click_time) > 0.4:
                    pyautogui.click()
                    state.add_event("Left Click", now)
                    hud.append("LEFT CLICK")
                    state.last_click_time = now
                    draw_neon_glow(frame, px[LM_INDEX_TIP], (0, 255, 0), 20)
            state.index_prev_y[label] = curr_y
            continue

    return hud

def handle_controls(
    frame: np.ndarray,
    hands: list[HandSnapshot],
    state: ControlState,
    now: float,
    screen_w: int,
    screen_h: int,
) -> list[str]:
    """Unified entry point for simple gesture set."""
    return detect_gestures(frame, hands, state, now, screen_w, screen_h)


HAND_CONNECTIONS = mp_hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS
 
def draw_text_with_bg(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale=0.6,
    color=(255, 255, 255),
    thickness=1,
    bg_color=(0, 0, 0),
    alpha=0.6
) -> None:
    """Professional HUD text with semi-transparent background box."""
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y - th - 5), (x + tw + 5, y + 5), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
 
def draw_neon_glow(frame: np.ndarray, point: tuple[int, int], color: tuple[int, int, int], radius: int) -> None:
    """Draws multi-layered transparent circles for a neon/glow look."""
    overlay = frame.copy()
    for r in range(radius, radius + 15, 3):
        alpha = 0.5 - (r - radius) / 30.0
        if alpha <= 0: break
        cv2.circle(overlay, point, r, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
 
def draw_premium_dashboard(frame: np.ndarray, detected_hands: list[str]) -> None:
    """Header bar with AI status and hand icons."""
    h, w = frame.shape[:2]
    # Header box
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Text indicators in top right
    right_on = "Right" in detected_hands
    left_on = "Left" in detected_hands
    
    # Right hand status
    rc = (0, 255, 0) if right_on else (100, 100, 100)
    draw_text_with_bg(frame, "RIGHT", (w - 180, 32), scale=0.45, color=rc)
    # Left hand status
    lc = (0, 255, 0) if left_on else (100, 100, 100)
    draw_text_with_bg(frame, "LEFT ", (w - 100, 32), scale=0.45, color=lc)


def main() -> None:
    """Open camera, run HandLandmarker loop, map hands to desktop control."""
    screen_w, screen_h = pyautogui.size()

    if sys.platform == "win32":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)

    model_path = ensure_hand_landmarker_model()
    hand_options = mp_hand_landmarker.HandLandmarkerOptions(
        base_options=mp_base_options.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision_mode.VisionTaskRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_timestamp_ms = 0
    control_state = ControlState()

    cv2.namedWindow("Unified Hand Control", cv2.WINDOW_NORMAL)

    with mp_hand_landmarker.HandLandmarker.create_from_options(hand_options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            now = time.monotonic()

            frame_rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_frame = MpImage(image_format=ImageFormat.SRGB, data=frame_rgb)

            frame_timestamp_ms += 33
            result = landmarker.detect_for_video(mp_frame, frame_timestamp_ms)

            hands_list: list[HandSnapshot] = []
            if result.hand_landmarks and result.handedness:
                for lm_list, handedness_list in zip(
                    result.hand_landmarks, result.handedness
                ):
                    hand_label = (
                        handedness_list[0].category_name
                        if handedness_list
                        else "Unknown"
                    )
                    hands_list.append(build_hand_snapshot(hand_label, lm_list, fw, fh))

            hud_events = handle_controls(
                frame, hands_list, control_state, now, screen_w, screen_h
            )

            if result.hand_landmarks:
                for lm_list in result.hand_landmarks:
                    draw_hand_connections_bgr(frame, lm_list, HAND_CONNECTIONS)
 
            # Premium Visual UI
            current_hand_labels = [h.label for h in hands_list]
            draw_premium_dashboard(frame, current_hand_labels)
 
            y0 = 85
            for msg in hud_events[:CONFIG.HISTORY_SIZE]:
                draw_text_with_bg(frame, f" > {msg}", (15, y0), color=(0, 255, 200), scale=0.5, bg_color=(20, 20, 20))
                y0 += 32
 
            # Update the window tab/title with the persistent history
            history_str = control_state.get_title_status(now)
            window_title = "Pro Hand Control"
            if history_str:
                window_title += f" [ {history_str} ]"
            else:
                window_title += " [ Ready ]"
 
            try:
                cv2.setWindowTitle("Unified Hand Control", window_title)
            except Exception:
                pass
 
            cv2.imshow("Unified Hand Control", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
