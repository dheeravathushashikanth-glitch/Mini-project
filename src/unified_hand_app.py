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
    """Centralized configuration for all thresholds and timings."""
    # Detection
    CLICK_THRESHOLD: int = 30
    PINCH_MAX_DIST: int = 38
    TWO_FINGER_TIP_MAX: int = 45
    
    # Volume Gestures
    INDEX_MIDDLE_VOL_DOWN_LO: int = 40
    INDEX_MIDDLE_VOL_DOWN_HI: int = 100
    INDEX_MIDDLE_VOL_UP_MIN: int = 100
    HAND_VOL_DELTA_MIN: int = 20
    
    # Navigation
    THUMB_MIDDLE_TAB_MIN: int = 150
    INDEX_PINKY_NEW_TAB_MIN: int = 180
    INDEX_PINKY_OPEN_NORM: float = 0.22
    SWIPE_ACCUM_PX: int = 120
    SWIPE_DECAY: float = 0.85
    
    # Scrolling
    SCROLL_STEP_ACCUM: int = 20
    PINCH_SCROLL_ACCUM: int = 28
    HSCROLL_ACCUM: int = 40
    MOBILE_SCROLL_DELTA_MIN: int = 8
    
    # Timings (seconds)
    GESTURE_COOLDOWN: float = 0.45
    TAB_HOTKEY_COOLDOWN: float = 0.55
    VOLUME_COOLDOWN: float = 0.12
    MOBILE_SCROLL_RESET: float = 0.2
    RIGHT_CLICK_COOLDOWN: float = 0.3
    CURSOR_SMOOTH_DURATION: float = 0.1
    POSE_HUD_COOLDOWN: float = 0.8
    WINDOW_TITLE_PERSISTENCE: float = 2.0
    
    # UI
    HISTORY_SIZE: int = 5

CONFIG = AppConfig()


class GestureId(IntEnum):
    """
    Unified pose identifiers: fist, isolated fingers, V-shape, two-finger cluster,
    and pinch-derived majors/minors handled separately in ``detect_gestures``.
    """

    FIST = 0
    PINKY = 1
    RING = 2
    MID = 4
    INDEX = 8
    V_GEST = 33
    TWO_FINGER_CLOSED = 34
    PINCH_MAJOR = 35
    PINCH_MINOR = 36
    NONE = 99


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
    d_thumb_middle: float
    d_index_pinky: float
    d_thumb_index: float
    pinch_mid_norm: tuple[float, float]
    gesture: GestureId
    midpoint_im_scroll: tuple[int, int]


def build_hand_snapshot(label: str, lm, w: int, h: int) -> HandSnapshot:
    px = [lm_to_px(lm[i], w, h) for i in range(21)]
    fs = finger_states(lm)
    d_im = dist_px(px[LM_INDEX_TIP], px[LM_MIDDLE_TIP])
    d_tm = dist_px(px[LM_THUMB_TIP], px[LM_MIDDLE_TIP])
    d_ipk = dist_px(px[LM_INDEX_TIP], px[LM_PINKY_TIP])
    d_ti = dist_px(px[LM_THUMB_TIP], px[LM_INDEX_TIP])
    mid_ix = (px[LM_INDEX_TIP][0] + px[LM_MIDDLE_TIP][0]) // 2
    mid_iy = (px[LM_INDEX_TIP][1] + px[LM_MIDDLE_TIP][1]) // 2
    g = classify_hand_pose(fs)
    if fs["index"] and fs["middle"] and d_im < CONFIG.TWO_FINGER_TIP_MAX:
        g = GestureId.TWO_FINGER_CLOSED
    pinch_mid = (
        (lm[LM_THUMB_TIP].x + lm[LM_INDEX_TIP].x) * 0.5,
        (lm[LM_THUMB_TIP].y + lm[LM_INDEX_TIP].y) * 0.5,
    )
    return HandSnapshot(
        label=label,
        lm=lm,
        px=px,
        fs=fs,
        d_index_middle=d_im,
        d_thumb_middle=d_tm,
        d_index_pinky=d_ipk,
        d_thumb_index=d_ti,
        pinch_mid_norm=pinch_mid,
        gesture=g,
        midpoint_im_scroll=(mid_ix, mid_iy),
    )


@dataclass
class ControlState:
    """
    Edge-triggered actions: ``held_actions`` clears when the condition goes False.
    Per-hand dicts use MediaPipe ``Left`` / ``Right`` labels.
    """

    held_actions: dict[str, bool] = field(default_factory=dict)
    last_fire_time: dict[str, float] = field(default_factory=dict)
    drag_armed: bool = False
    pinch_mid_norm: dict[str, tuple[float, float]] = field(default_factory=dict)
    pinch_major_acc: dict[str, float] = field(default_factory=dict)
    pinch_minor_acc: dict[str, float] = field(default_factory=dict)
    scroll_last_im: dict[str, tuple[int, int]] = field(default_factory=dict)
    scroll_acc: dict[str, float] = field(default_factory=dict)
    swipe_vel_x: float = 0.0
    swipe_vel_y: float = 0.0
    prev_index_tip: tuple[int, int] | None = None

    # New Mobile Scroll state
    scroll_flag: bool = False
    mobile_scroll_last_time: float = 0.0
    prev_index_y: float | None = None
    scroll_direction: int = 0

    # New Right Click state (Pinch)
    right_click_flag: bool = False
    right_click_last_time: float = 0.0

    # New Volume Control state (Whole Hand)
    volume_flag: bool = False
    volume_last_time: float = 0.0
    prev_avg_hand_y: float | None = None
    volume_direction: int = 0

    # Gesture History tracking
    history: deque[str] = field(default_factory=lambda: deque(maxlen=CONFIG.HISTORY_SIZE))
    last_event_time: float = 0.0

    def add_event(self, event: str, now: float) -> None:
        """Adds a gesture event to the history and updates the timestamp."""
        if not self.history or self.history[-1] != event:
            self.history.append(event)
        self.last_event_time = now

    def get_title_status(self, now: float) -> str:
        """Returns a string representing the recent gesture history."""
        if not self.history:
            return ""
        # If quiet for a while, we could potentially clear or fade, 
        # but the user asked to 'keep it for a while'.
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
    """
    All non-cursor discrete/continuous gestures: fist drag, pinch axes, two-finger scroll,
    volume, tab hotkeys, optional single-finger HUD labels.
    """
    hud: list[str] = []

    any_fist = any(h.gesture == GestureId.FIST for h in hands)
    if any_fist:
        if not state.drag_armed:
            pyautogui.mouseDown()
            state.drag_armed = True
            hud.append("FIST: drag")
    else:
        if state.drag_armed:
            pyautogui.mouseUp()
            state.drag_armed = False

    active_two_finger_labels: set[str] = set()

    for hs in hands:
        label = hs.label
        px = hs.px
        pinched = hs.d_thumb_index < CONFIG.PINCH_MAX_DIST

        # Mobile Scroll Gesture (Right Hand Only)
        if label == "Right":
            current_y = px[LM_INDEX_TIP][1]
            
            # Reset flag after timeout
            if state.scroll_flag and (now - state.mobile_scroll_last_time) > CONFIG.MOBILE_SCROLL_RESET:
                state.scroll_flag = False
                state.scroll_direction = 0
                
            if state.prev_index_y is not None:
                delta_y = current_y - state.prev_index_y
                if abs(delta_y) > CONFIG.MOBILE_SCROLL_DELTA_MIN and not state.scroll_flag:
                    if delta_y < -CONFIG.MOBILE_SCROLL_DELTA_MIN:  # Moving UP
                        pyautogui.scroll(3)
                        state.scroll_direction = 1
                        state.add_event("Scroll UP", now)
                    elif delta_y > CONFIG.MOBILE_SCROLL_DELTA_MIN:  # Moving DOWN
                        pyautogui.scroll(-3)
                        state.scroll_direction = -1
                        state.add_event("Scroll DOWN", now)
                    
                    state.scroll_flag = True
                    state.mobile_scroll_last_time = now
            
            state.prev_index_y = current_y
            
            # Visual feedback
            if state.scroll_direction != 0:
                glow_color = (0, 0, 255) if state.scroll_direction == 1 else (255, 0, 0)
                draw_neon_glow(frame, px[LM_INDEX_TIP], glow_color, 15)
                tip_x, tip_y = px[LM_INDEX_TIP]
                # Blue arrow showing scroll direction (more robust)
                arrow_c = (255, 128, 0)
                if state.scroll_direction == 1:
                    cv2.arrowedLine(frame, (tip_x, tip_y + 45), (tip_x, tip_y - 15), arrow_c, 4, tipLength=0.3)
                    hud.append("Right: Scroll UP")
                elif state.scroll_direction == -1:
                    cv2.arrowedLine(frame, (tip_x, tip_y - 45), (tip_x, tip_y + 15), arrow_c, 4, tipLength=0.3)
                    hud.append("Right: Scroll DOWN")

            # --- Right Click Gesture (Pinch) ---
            dist_ti = hs.d_thumb_index
            if state.right_click_flag and (now - state.right_click_last_time) > CONFIG.RIGHT_CLICK_COOLDOWN:
                state.right_click_flag = False
                
            if dist_ti < 40 and not state.right_click_flag:
                pyautogui.rightClick()
                state.right_click_flag = True
                state.right_click_last_time = now
                state.add_event("Right Click", now)
                
            if dist_ti < 40:
                # Visual: Yellow line
                cv2.line(frame, px[LM_THUMB_TIP], px[LM_INDEX_TIP], (0, 255, 255), 2)
                hud.append("Right: Right Click")
                
            # --- Volume Control Gesture (Whole Hand Movement) ---
            wrist_x, wrist_y = px[LM_WRIST]
            middle_y = px[LM_MIDDLE_TIP][1]
            current_avg_y = (wrist_y + middle_y) / 2.0
            
            if state.volume_flag and (now - state.volume_last_time) > CONFIG.VOLUME_COOLDOWN:
                state.volume_flag = False
                state.volume_direction = 0
                
            if state.prev_avg_hand_y is not None:
                delta_y = current_avg_y - state.prev_avg_hand_y
                if abs(delta_y) > CONFIG.HAND_VOL_DELTA_MIN and not state.volume_flag:
                    if delta_y < -CONFIG.HAND_VOL_DELTA_MIN:  # Hand moves UP
                        pyautogui.press("volumeup")
                        state.volume_direction = 1
                        state.add_event("VOL UP", now)
                    elif delta_y > CONFIG.HAND_VOL_DELTA_MIN:  # Hand moves DOWN
                        pyautogui.press("volumedown")
                        state.volume_direction = -1
                        state.add_event("VOL DOWN", now)
                        
                    state.volume_flag = True
                    state.volume_last_time = now
            
            state.prev_avg_hand_y = current_avg_y
            
            # Visual: Green circle and Purple arrow
            if state.volume_direction != 0:
                draw_neon_glow(frame, px[LM_WRIST], (0, 255, 128), 20)
                if state.volume_direction == 1:
                    cv2.arrowedLine(frame, (wrist_x, wrist_y + 40), (wrist_x, wrist_y - 40), (255, 0, 255), 4, tipLength=0.3)
                    hud.append("Right: Vol UP")
                elif state.volume_direction == -1:
                    cv2.arrowedLine(frame, (wrist_x, wrist_y - 40), (wrist_x, wrist_y + 40), (255, 0, 255), 4, tipLength=0.3)
                    hud.append("Right: Vol DOWN")

        if pinched:
            prev = state.pinch_mid_norm.get(label)
            state.pinch_mid_norm[label] = hs.pinch_mid_norm
            if prev is not None:
                dx = (hs.pinch_mid_norm[0] - prev[0]) * screen_w
                dy = (hs.pinch_mid_norm[1] - prev[1]) * screen_h
                if abs(dy) > abs(dx) + 5:
                    state.pinch_minor_acc[label] = 0.0
                    state.pinch_major_acc[label] = (
                        state.pinch_major_acc.get(label, 0.0) + dy
                    )
                    acc_y = state.pinch_major_acc[label]
                    while acc_y > CONFIG.PINCH_SCROLL_ACCUM:
                        pyautogui.scroll(-1)
                        acc_y -= CONFIG.PINCH_SCROLL_ACCUM
                        hud.append(f"{label}: PINCH_MAJOR ↓")
                        state.add_event(f"{label} Scroll ↓", now)
                    while acc_y < -CONFIG.PINCH_SCROLL_ACCUM:
                        pyautogui.scroll(1)
                        acc_y += CONFIG.PINCH_SCROLL_ACCUM
                        hud.append(f"{label}: PINCH_MAJOR ↑")
                        state.add_event(f"{label} Scroll ↑", now)
                    state.pinch_major_acc[label] = acc_y
                elif abs(dx) > abs(dy) + 5:
                    state.pinch_major_acc[label] = 0.0
                    state.pinch_minor_acc[label] = (
                        state.pinch_minor_acc.get(label, 0.0) + dx
                    )
                    acc_x = state.pinch_minor_acc[label]
                    while acc_x > CONFIG.HSCROLL_ACCUM:
                        try:
                            pyautogui.hscroll(1)
                        except AttributeError:
                            pyautogui.keyDown("shift")
                            pyautogui.scroll(-1)
                            pyautogui.keyUp("shift")
                        acc_x -= CONFIG.HSCROLL_ACCUM
                        hud.append(f"{label}: PINCH_MINOR →")
                        state.add_event(f"{label} H-Scroll →", now)
                    while acc_x < -CONFIG.HSCROLL_ACCUM:
                        try:
                            pyautogui.hscroll(-1)
                        except AttributeError:
                            pyautogui.keyDown("shift")
                            pyautogui.scroll(1)
                            pyautogui.keyUp("shift")
                        acc_x += CONFIG.HSCROLL_ACCUM
                        hud.append(f"{label}: PINCH_MINOR ←")
                        state.add_event(f"{label} H-Scroll ←", now)
                    state.pinch_minor_acc[label] = acc_x
            draw_green_pair(frame, px[LM_THUMB_TIP], px[LM_INDEX_TIP])
        else:
            state.pinch_mid_norm.pop(label, None)
            state.pinch_major_acc.pop(label, None)
            state.pinch_minor_acc.pop(label, None)

        if hs.gesture == GestureId.TWO_FINGER_CLOSED and not pinched:
            active_two_finger_labels.add(label)
            draw_green_pair(frame, px[LM_INDEX_TIP], px[LM_MIDDLE_TIP])
            cy = hs.midpoint_im_scroll[1]
            prev_im = state.scroll_last_im.get(label)
            state.scroll_last_im[label] = hs.midpoint_im_scroll
            if prev_im is not None:
                dy = cy - prev_im[1]
                acc = state.scroll_acc.get(label, 0.0) + dy
                while acc > CONFIG.SCROLL_STEP_ACCUM:
                    pyautogui.scroll(-1)
                    acc -= CONFIG.SCROLL_STEP_ACCUM
                    hud.append(f"{label}: 2-finger ↓")
                    state.add_event(f"{label} 2-finger ↓", now)
                while acc < -CONFIG.SCROLL_STEP_ACCUM:
                    pyautogui.scroll(1)
                    acc += CONFIG.SCROLL_STEP_ACCUM
                    hud.append(f"{label}: 2-finger ↑")
                    state.add_event(f"{label} 2-finger ↑", now)
                state.scroll_acc[label] = acc

        vol_pose = (
            hs.fs["index"]
            and hs.fs["middle"]
            and hs.gesture != GestureId.TWO_FINGER_CLOSED
            and not pinched
        )
        if vol_pose:
            draw_green_pair(frame, px[LM_INDEX_TIP], px[LM_MIDDLE_TIP])
            in_down = CONFIG.INDEX_MIDDLE_VOL_DOWN_LO < hs.d_index_middle < CONFIG.INDEX_MIDDLE_VOL_DOWN_HI
            in_up = hs.d_index_middle > CONFIG.INDEX_MIDDLE_VOL_UP_MIN

            def _vd() -> None:
                pyautogui.press("volumedown")

            def _vu() -> None:
                pyautogui.press("volumeup")

            if pulse_action(state, f"vd_{label}", in_down, now, CONFIG.VOLUME_COOLDOWN, _vd):
                hud.append(f"{label}: Vol Down")
                state.add_event(f"{label} VOL-", now)
            if pulse_action(state, f"vu_{label}", in_up, now, CONFIG.VOLUME_COOLDOWN, _vu):
                hud.append(f"{label}: Vol Up")
                state.add_event(f"{label} VOL+", now)
        else:
            _held_reset(state, f"vd_{label}")
            _held_reset(state, f"vu_{label}")

        tab_cond = (
            hs.fs["middle"]
            and hs.d_thumb_middle > CONFIG.THUMB_MIDDLE_TAB_MIN
            and not pinched
        )
        if tab_cond:
            draw_green_pair(frame, px[LM_THUMB_TIP], px[LM_MIDDLE_TIP])

            def _ctab() -> None:
                pyautogui.hotkey("ctrl", "tab")

            if pulse_action(
                state, f"ctab_{label}", True, now, CONFIG.TAB_HOTKEY_COOLDOWN, _ctab
            ):
                hud.append(f"{label}: Ctrl+Tab")
                state.add_event("Next Tab", now)
        else:
            _held_reset(state, f"ctab_{label}")

        d_ip_norm = dist_norm(hs.lm[LM_INDEX_TIP], hs.lm[LM_PINKY_TIP])
        new_tab_cond = (
            hs.fs["index"]
            and hs.fs["pinky"]
            and (
                hs.d_index_pinky > CONFIG.INDEX_PINKY_NEW_TAB_MIN
                or d_ip_norm > CONFIG.INDEX_PINKY_OPEN_NORM
            )
            and not pinched
        )
        if new_tab_cond:
            draw_green_pair(frame, px[LM_INDEX_TIP], px[LM_PINKY_TIP])

            def _newt() -> None:
                pyautogui.hotkey("ctrl", "t")

            if pulse_action(
                state, f"newt_{label}", True, now, CONFIG.TAB_HOTKEY_COOLDOWN, _newt
            ):
                hud.append(f"{label}: Ctrl+T")
                state.add_event("New Tab", now)
        else:
            _held_reset(state, f"newt_{label}")

        if (
            hs.gesture
            in (GestureId.PINKY, GestureId.RING, GestureId.MID, GestureId.INDEX)
            and not pinched
            and hs.gesture != GestureId.FIST
        ):

            def _noop() -> None:
                return

            if pulse_action(
                state,
                f"pose_{hs.gesture.name}_{label}",
                True,
                now,
                CONFIG.POSE_HUD_COOLDOWN,
                _noop,
            ):
                hud.append(f"{label}: {hs.gesture.name}")
                state.add_event(f"Pose: {hs.gesture.name}", now)

    for lbl in list(state.scroll_last_im.keys()):
        if lbl not in active_two_finger_labels:
            state.scroll_last_im.pop(lbl, None)
            state.scroll_acc.pop(lbl, None)
            
    if "Right" not in [h.label for h in hands]:
        state.prev_index_y = None
        state.scroll_direction = 0
        state.scroll_flag = False
        state.prev_avg_hand_y = None
        state.volume_direction = 0
        state.volume_flag = False
        state.right_click_flag = False

    return hud


def handle_controls(
    frame: np.ndarray,
    hands: list[HandSnapshot],
    state: ControlState,
    now: float,
    screen_w: int,
    screen_h: int,
) -> list[str]:
    """Run ``detect_gestures``, then swipe navigation, then pointer + button roles."""
    hud = detect_gestures(frame, hands, state, now, screen_w, screen_h)

    if hands:
        avg_ix = int(sum(h.px[LM_INDEX_TIP][0] for h in hands) / len(hands))
        avg_iy = int(sum(h.px[LM_INDEX_TIP][1] for h in hands) / len(hands))
        if state.prev_index_tip is None:
            state.prev_index_tip = (avg_ix, avg_iy)
        else:
            pxv = avg_ix - state.prev_index_tip[0]
            pyv = avg_iy - state.prev_index_tip[1]
            state.prev_index_tip = (avg_ix, avg_iy)
            state.swipe_vel_x = state.swipe_vel_x * CONFIG.SWIPE_DECAY + pxv
            state.swipe_vel_y = state.swipe_vel_y * CONFIG.SWIPE_DECAY + pyv

        want_back = state.swipe_vel_x < -CONFIG.SWIPE_ACCUM_PX
        want_fwd = state.swipe_vel_x > CONFIG.SWIPE_ACCUM_PX

        def _back() -> None:
            pyautogui.hotkey("alt", "left")

        def _fwd() -> None:
            pyautogui.hotkey("alt", "right")

        if pulse_action(state, "swipe_back", want_back, now, CONFIG.TAB_HOTKEY_COOLDOWN, _back):
            hud.append("Swipe: Back")
            state.add_event("Swipe BACK", now)
            state.swipe_vel_x = 0.0
        if pulse_action(state, "swipe_fwd", want_fwd, now, CONFIG.TAB_HOTKEY_COOLDOWN, _fwd):
            hud.append("Swipe: Forward")
            state.add_event("Swipe FORWARD", now)
            state.swipe_vel_x = 0.0
    else:
        state.prev_index_tip = None
        state.swipe_vel_x *= CONFIG.SWIPE_DECAY
        state.swipe_vel_y *= CONFIG.SWIPE_DECAY

    for hs in hands:
        label = hs.label
        px = hs.px
        lm = hs.lm
        pinched = hs.d_thumb_index < CONFIG.PINCH_MAX_DIST

        if label == "Right":
            if hs.gesture == GestureId.V_GEST and not pinched:
                t8 = lm[LM_INDEX_TIP]
                sx = int(t8.x * screen_w)
                sy = int(t8.y * screen_h)
                pyautogui.moveTo(sx, sy, duration=CONFIG.CURSOR_SMOOTH_DURATION)
                cv2.circle(frame, px[LM_INDEX_TIP], 10, (255, 128, 0), 2)
            elif hs.fs["index"] and hs.gesture != GestureId.FIST and not pinched:
                t8 = lm[LM_INDEX_TIP]
                sx = int(t8.x * screen_w)
                sy = int(t8.y * screen_h)
                pyautogui.moveTo(sx, sy, duration=CONFIG.CURSOR_SMOOTH_DURATION * 0.5)
                cv2.circle(frame, px[LM_INDEX_TIP], 8, (255, 0, 0), -1)

        if label == "Left":
            thumb = px[LM_THUMB_TIP]
            index = px[LM_INDEX_TIP]
            index2 = px[LM_INDEX_PIP]
            pinky = px[LM_PINKY_TIP]
            ds = dist_px(thumb, index)
            dd = dist_px(thumb, index2)
            dr = dist_px(thumb, pinky)

            if ds < CONFIG.CLICK_THRESHOLD:

                def _lc() -> None:
                    pyautogui.click()

                if pulse_action(state, "left_click", True, now, CONFIG.GESTURE_COOLDOWN, _lc):
                    hud.append("Left: Click")
                    state.add_event("Left CLICK", now)
            else:
                _held_reset(state, "left_click")

            if dd < CONFIG.CLICK_THRESHOLD:

                def _dc() -> None:
                    pyautogui.doubleClick()

                if pulse_action(state, "dbl_click", True, now, CONFIG.GESTURE_COOLDOWN, _dc):
                    hud.append("Left: Double")
                    state.add_event("Left DOUBLE", now)
            else:
                _held_reset(state, "dbl_click")

            if dr < CONFIG.CLICK_THRESHOLD:

                def _rc() -> None:
                    pyautogui.rightClick()

                if pulse_action(state, "right_click", True, now, CONFIG.GESTURE_COOLDOWN, _rc):
                    hud.append("Left: Right")
                    state.add_event("Left RIGHT-CLK", now)
            else:
                _held_reset(state, "right_click")

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
    
    # Text
    cv2.putText(frame, "PRO AI GESTURE CENTER v2.0", (15, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 255), 1, cv2.LINE_AA)
    
    # Hand indicators in top right
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
