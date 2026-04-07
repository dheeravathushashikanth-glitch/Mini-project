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
    """Zero-collision 10-function gesture configuration."""
    # Touch thresholds (pixels)
    INDEX_THUMB_TOUCH: int = 38    # Left Click: Index tip + Thumb tip
    MIDDLE_THUMB_TOUCH: int = 38   # Right Click: Middle tip + Thumb tip

    # Motion thresholds (pixels per frame)
    SCROLL_DELTA_Y: int = 12       # Scroll: vertical movement of Index+Middle
    SWITCH_TAB_DELTA_X: int = 30   # Switch Tab: horizontal swipe of Index+Middle
    SWITCH_TAB_ACCUM: int = 60     # Accumulate px before firing tab switch

    # Cursor smoothing
    CURSOR_SMOOTH_DURATION: float = 0.06

    # Cooldowns (seconds)
    CLICK_COOLDOWN: float = 0.4
    VOLUME_COOLDOWN: float = 0.35
    TAB_COOLDOWN: float = 0.5
    NEW_TAB_COOLDOWN: float = 1.0
    SCROLL_COOLDOWN: float = 0.08

    # UI
    HISTORY_SIZE: int = 5
    WINDOW_TITLE_PERSISTENCE: float = 2.0

CONFIG = AppConfig()


class GestureId(IntEnum):
    """Finger-count based gesture IDs for the 10-function zero-collision system."""
    FIST = 0          # 0 fingers: Pause/Resume
    INDEX_ONLY = 1    # 1 finger (index): Cursor Move  
    TWO_FINGERS = 2   # 2 fingers (index+middle): Scroll / Switch Tab
    THREE_FINGERS = 3 # 3 fingers (index+middle+ring): Volume Up
    FOUR_FINGERS = 4  # 4 fingers (no thumb): Volume Down
    OPEN_PALM = 5     # 5 fingers: Open New Tab
    OTHER = 9         # Anything else (thumb alone, etc.)


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
    """Zero-collision classifier: purely based on which fingers are extended."""
    idx  = fs["index"]
    mid  = fs["middle"]
    ring = fs["ring"]
    pinky= fs["pinky"]
    thumb= fs["thumb"]

    n_fingers = sum([idx, mid, ring, pinky])  # thumb excluded from count

    if n_fingers == 0 and not thumb:  return GestureId.FIST
    if n_fingers == 1 and idx:        return GestureId.INDEX_ONLY
    if n_fingers == 2 and idx and mid and not ring and not pinky: return GestureId.TWO_FINGERS
    if n_fingers == 3 and idx and mid and ring and not pinky:     return GestureId.THREE_FINGERS
    if n_fingers == 4 and idx and mid and ring and pinky:         return GestureId.FOUR_FINGERS
    if n_fingers == 4 and thumb and idx and mid and ring and pinky: return GestureId.OPEN_PALM
    if thumb and idx and mid and ring and pinky:                  return GestureId.OPEN_PALM
    return GestureId.OTHER


@dataclass
class HandSnapshot:
    label: str
    lm: list
    px: list[tuple[int, int]]
    fs: dict[str, bool]
    d_thumb_index: float   # Index + Thumb distance  -> Left Click
    d_thumb_middle: float  # Middle + Thumb distance -> Right Click
    midpoint_im: tuple[int, int]  # Midpoint of Index+Middle for scroll/tab tracking
    gesture: GestureId


def build_hand_snapshot(label: str, lm, w: int, h: int) -> HandSnapshot:
    px = [lm_to_px(lm[i], w, h) for i in range(21)]
    fs = finger_states(lm)
    d_ti  = dist_px(px[LM_THUMB_TIP], px[LM_INDEX_TIP])
    d_tm  = dist_px(px[LM_THUMB_TIP], px[LM_MIDDLE_TIP])
    mid_x = (px[LM_INDEX_TIP][0] + px[LM_MIDDLE_TIP][0]) // 2
    mid_y = (px[LM_INDEX_TIP][1] + px[LM_MIDDLE_TIP][1]) // 2
    g = classify_hand_pose(fs)
    return HandSnapshot(
        label=label,
        lm=lm,
        px=px,
        fs=fs,
        d_thumb_index=d_ti,
        d_thumb_middle=d_tm,
        midpoint_im=(mid_x, mid_y),
        gesture=g,
    )


@dataclass
class ControlState:
    """State for zero-collision 10-function gesture system."""

    # Cooldown tracking
    last_fire_time: dict[str, float] = field(default_factory=dict)

    # Two-finger midpoint tracking for Scroll and Switch Tab
    prev_two_finger_mid: tuple[int, int] | None = None
    tab_swipe_acc: float = 0.0

    # System pause toggle (Closed Fist)
    system_paused: bool = False
    fist_fired: bool = False  # so fist only toggles once per hold

    # Gesture history for HUD
    history: deque[str] = field(default_factory=lambda: deque(maxlen=CONFIG.HISTORY_SIZE))
    last_event_time: float = 0.0

    def cooldown_ok(self, key: str, now: float, cooldown: float) -> bool:
        """Returns True if enough time has passed since last fire for this key."""
        if now - self.last_fire_time.get(key, 0.0) >= cooldown:
            self.last_fire_time[key] = now
            return True
        return False

    def add_event(self, event: str, now: float) -> None:
        if not self.history or self.history[-1] != event:
            self.history.append(event)
        self.last_event_time = now

    def get_title_status(self, now: float) -> str:
        if not self.history:
            return ""
        return " | ".join(list(self.history))


def draw_green_pair(frame: np.ndarray, p: tuple[int, int], q: tuple[int, int]) -> None:
    """Green segment between fingertips."""
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


def handle_controls(
    frame: np.ndarray,
    hands: list[HandSnapshot],
    state: ControlState,
    now: float,
    screen_w: int,
    screen_h: int,
) -> list[str]:
    """
    Zero-collision 10-function gesture dispatcher.
    Priority order (highest first):
      FIST -> OPEN_PALM -> FOUR_FINGERS -> THREE_FINGERS -> TWO_FINGERS -> INDEX_ONLY -> TOUCH checks
    Only the FIRST matched gesture block executes per hand per frame.
    """
    hud: list[str] = []

    if not hands:
        state.prev_two_finger_mid = None
        state.tab_swipe_acc = 0.0
        state.fist_fired = False
        return hud

    # Use the first visible hand (whichever is detected)
    hs = hands[0]
    px = hs.px
    lm = hs.lm
    fs = hs.fs
    g  = hs.gesture

    # ── GESTURE 10: PAUSE / RESUME (Closed Fist) ────────────────────────────
    if g == GestureId.FIST:
        if not state.fist_fired:
            state.system_paused = not state.system_paused
            state.fist_fired = True
            label = "PAUSED" if state.system_paused else "RESUMED"
            state.add_event(label, now)
            hud.append(label)
            draw_neon_glow(frame, px[LM_WRIST], (0, 0, 180), 25)
        return hud
    else:
        state.fist_fired = False

    # Stop everything if paused
    if state.system_paused:
        draw_text_with_bg(frame, "  SYSTEM PAUSED  ", (10, 120),
                          scale=0.9, color=(0, 0, 255), bg_color=(20, 20, 20))
        return hud

    # ── GESTURE 9: OPEN NEW TAB (All 5 fingers / Open Palm) ─────────────────
    if g == GestureId.OPEN_PALM:
        draw_neon_glow(frame, px[LM_WRIST], (0, 220, 255), 22)
        if state.cooldown_ok("new_tab", now, CONFIG.NEW_TAB_COOLDOWN):
            pyautogui.hotkey("ctrl", "t")
            state.add_event("New Tab", now)
            hud.append("NEW TAB")
        state.prev_two_finger_mid = None
        return hud

    # ── GESTURE 8: VOLUME DOWN (4 fingers up, no thumb) ─────────────────────
    if g == GestureId.FOUR_FINGERS:
        draw_neon_glow(frame, px[LM_MIDDLE_TIP], (0, 100, 255), 18)
        if state.cooldown_ok("vol_down", now, CONFIG.VOLUME_COOLDOWN):
            pyautogui.press("volumedown")
            state.add_event("Vol DOWN", now)
            hud.append("VOL DOWN")
        state.prev_two_finger_mid = None
        return hud

    # ── GESTURE 7: VOLUME UP (Index + Middle + Ring, 3 fingers) ─────────────
    if g == GestureId.THREE_FINGERS:
        draw_neon_glow(frame, px[LM_MIDDLE_TIP], (0, 255, 128), 18)
        if state.cooldown_ok("vol_up", now, CONFIG.VOLUME_COOLDOWN):
            pyautogui.press("volumeup")
            state.add_event("Vol UP", now)
            hud.append("VOL UP")
        state.prev_two_finger_mid = None
        return hud

    # ── GESTURE 5 & 6: SCROLL UP/DOWN + SWITCH TABS (Index + Middle) ────────
    if g == GestureId.TWO_FINGERS:
        draw_green_pair(frame, px[LM_INDEX_TIP], px[LM_MIDDLE_TIP])
        mid = hs.midpoint_im
        prev = state.prev_two_finger_mid
        state.prev_two_finger_mid = mid

        if prev is not None:
            dy = mid[1] - prev[1]   # positive = moving down
            dx = mid[0] - prev[0]   # positive = moving right

            # Prioritise whichever axis is dominant
            if abs(dy) > abs(dx):   # ── Vertical → Scroll
                state.tab_swipe_acc = 0.0
                if dy < -CONFIG.SCROLL_DELTA_Y and state.cooldown_ok("scroll_up", now, CONFIG.SCROLL_COOLDOWN):
                    pyautogui.scroll(3)
                    state.add_event("Scroll UP", now)
                    hud.append("SCROLL UP")
                    draw_neon_glow(frame, mid, (0, 0, 255), 14)
                elif dy > CONFIG.SCROLL_DELTA_Y and state.cooldown_ok("scroll_dn", now, CONFIG.SCROLL_COOLDOWN):
                    pyautogui.scroll(-3)
                    state.add_event("Scroll DOWN", now)
                    hud.append("SCROLL DOWN")
                    draw_neon_glow(frame, mid, (255, 80, 0), 14)
            else:                   # ── Horizontal → Switch Tab
                state.tab_swipe_acc += dx
                if state.tab_swipe_acc > CONFIG.SWITCH_TAB_ACCUM:
                    if state.cooldown_ok("tab_next", now, CONFIG.TAB_COOLDOWN):
                        pyautogui.hotkey("ctrl", "tab")
                        state.add_event("Next Tab", now)
                        hud.append("NEXT TAB →")
                    state.tab_swipe_acc = 0.0
                elif state.tab_swipe_acc < -CONFIG.SWITCH_TAB_ACCUM:
                    if state.cooldown_ok("tab_prev", now, CONFIG.TAB_COOLDOWN):
                        pyautogui.hotkey("ctrl", "shift", "tab")
                        state.add_event("Prev Tab", now)
                        hud.append("PREV TAB ←")
                    state.tab_swipe_acc = 0.0
        return hud
    else:
        state.prev_two_finger_mid = None
        state.tab_swipe_acc = 0.0

    # ── GESTURE 1: CURSOR MOVE (Index finger only) ───────────────────────────
    if g == GestureId.INDEX_ONLY:
        # Check touches FIRST before moving cursor

        # GESTURE 2: LEFT CLICK — Index + Thumb touch
        if hs.d_thumb_index < CONFIG.INDEX_THUMB_TOUCH:
            draw_green_pair(frame, px[LM_THUMB_TIP], px[LM_INDEX_TIP])
            if state.cooldown_ok("left_click", now, CONFIG.CLICK_COOLDOWN):
                pyautogui.click()
                state.add_event("Left Click", now)
                hud.append("LEFT CLICK")
                draw_neon_glow(frame, px[LM_INDEX_TIP], (0, 255, 0), 20)
            return hud

        # Pure cursor movement
        t8 = lm[LM_INDEX_TIP]
        sx = int(t8.x * screen_w)
        sy = int(t8.y * screen_h)
        pyautogui.moveTo(sx, sy, duration=CONFIG.CURSOR_SMOOTH_DURATION)
        cv2.circle(frame, px[LM_INDEX_TIP], 10, (255, 200, 0), -1)
        return hud

    # ── GESTURE 3: RIGHT CLICK — Middle + Thumb touch (any other pose) ──────
    if hs.d_thumb_middle < CONFIG.MIDDLE_THUMB_TOUCH:
        draw_green_pair(frame, px[LM_THUMB_TIP], px[LM_MIDDLE_TIP])
        if state.cooldown_ok("right_click", now, CONFIG.CLICK_COOLDOWN):
            pyautogui.rightClick()
            state.add_event("Right Click", now)
            hud.append("RIGHT CLICK")
            draw_neon_glow(frame, px[LM_MIDDLE_TIP], (0, 255, 255), 20)
        return hud

    return hud





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
