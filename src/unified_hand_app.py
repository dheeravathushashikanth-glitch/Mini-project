"""
Unified hand-gesture desktop control — Zero-Collision 10-Function Engine.

Uses MediaPipe HandLandmarker (Tasks API), OpenCV, PyAutoGUI.
Forked from: https://github.com/varshithdharmaj/hand-gesture-virtual-mouse

Features:
  • Zero-collision 10-function gesture set (finger-count based)
  • Neon futuristic HUD with scanline overlay and animated borders
  • Persistent gesture history stack panel (right-side drawer)
  • Window stays always-on-top across all gestures (no focus steal)
  • Centralized AppConfig for easy threshold tuning
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

import cv2
import numpy as np
import pyautogui

from mediapipe.tasks.python.core import base_options as mp_base_options
from mediapipe.tasks.python.vision import hand_landmarker as mp_hand_landmarker
from mediapipe.tasks.python.vision.core.image import Image as MpImage
from mediapipe.tasks.python.vision.core.image import ImageFormat
from mediapipe.tasks.python.vision.core import vision_task_running_mode as mp_vision_mode

# ─────────────────────────────────────────────────────────────────────────────
# Model download
# ─────────────────────────────────────────────────────────────────────────────
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILENAME = "hand_landmarker.task"

# ─────────────────────────────────────────────────────────────────────────────
# Landmark indices (MediaPipe 21-point topology)
# ─────────────────────────────────────────────────────────────────────────────
LM_WRIST      = 0
LM_THUMB_TIP  = 4;  LM_THUMB_IP   = 3
LM_INDEX_TIP  = 8;  LM_INDEX_PIP  = 6;  LM_INDEX_MCP  = 5
LM_MIDDLE_TIP = 12; LM_MIDDLE_PIP = 10; LM_MIDDLE_MCP = 9
LM_RING_TIP   = 16; LM_RING_PIP   = 14
LM_PINKY_TIP  = 20; LM_PINKY_PIP  = 18

# ─────────────────────────────────────────────────────────────────────────────
# Neon colour palette  (BGR)
# ─────────────────────────────────────────────────────────────────────────────
CLR_CYAN    = (255, 220,   0)   # electric cyan
CLR_MAGENTA = (255,   0, 200)   # hot magenta
CLR_GREEN   = (  0, 255, 100)   # matrix green
CLR_ORANGE  = (  0, 140, 255)   # neon orange
CLR_PURPLE  = (200,   0, 200)   # violet
CLR_BLUE    = (255,  80,   0)   # cobalt blue
CLR_RED     = (  0,   0, 255)   # laser red
CLR_YELLOW  = (  0, 220, 255)   # neon yellow
CLR_WHITE   = (240, 240, 240)

# Gesture → neon colour mapping
GESTURE_COLORS: dict[str, tuple] = {
    "Left Click":  CLR_GREEN,
    "Right Click": CLR_CYAN,
    "Scroll UP":   CLR_BLUE,
    "Scroll DOWN": CLR_ORANGE,
    "Vol UP":      CLR_GREEN,
    "Vol DOWN":    CLR_RED,
    "Next Tab":    CLR_YELLOW,
    "Prev Tab":    CLR_YELLOW,
    "New Tab":     CLR_MAGENTA,
    "PAUSED":      CLR_RED,
    "RESUMED":     CLR_GREEN,
}

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (frozen → immutable at runtime)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AppConfig:
    """Zero-collision 10-function gesture configuration."""
    # Touch thresholds (pixels)
    INDEX_THUMB_TOUCH:  int   = 38
    MIDDLE_THUMB_TOUCH: int   = 38

    # Motion thresholds (pixels per frame)
    SCROLL_DELTA_Y:  int = 12
    SWITCH_TAB_ACCUM: int = 60

    # Cursor smoothing
    CURSOR_SMOOTH_DURATION: float = 0.06

    # Cooldowns (seconds)
    CLICK_COOLDOWN:  float = 0.40
    VOLUME_COOLDOWN: float = 0.35
    TAB_COOLDOWN:    float = 0.50
    NEW_TAB_COOLDOWN: float = 1.0
    SCROLL_COOLDOWN: float = 0.08

    # History panel
    HISTORY_SIZE:    int = 12    # max entries in the stack panel
    HISTORY_PANEL_W: int = 220   # width of the right-side history drawer
    HISTORY_FADE_S:  float = 8.0 # seconds before entry fully fades

CONFIG = AppConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Gesture IDs
# ─────────────────────────────────────────────────────────────────────────────
class GestureId(IntEnum):
    FIST         = 0   # ✊ pause / resume
    INDEX_ONLY   = 1   # ☝️  cursor / left-click
    TWO_FINGERS  = 2   # ✌️  scroll / switch tab
    THREE_FINGERS= 3   # 🤟 volume up
    FOUR_FINGERS = 4   # 🖐  volume down
    OPEN_PALM    = 5   # 🖐  new tab
    OTHER        = 9


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────
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
    print(f"Downloading {MODEL_FILENAME} (first run only)…")
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
    return dist_norm(lm[tip_i], lm[wrist_i]) > dist_norm(lm[pip_i], lm[wrist_i]) * 1.18

def thumb_extended(lm) -> bool:
    return dist_norm(lm[LM_THUMB_TIP], lm[LM_WRIST]) > dist_norm(lm[LM_THUMB_IP], lm[LM_WRIST]) * 1.12

def finger_states(lm) -> dict[str, bool]:
    return {
        "thumb":  thumb_extended(lm),
        "index":  finger_extended_from_wrist(lm, LM_INDEX_TIP,  LM_INDEX_PIP),
        "middle": finger_extended_from_wrist(lm, LM_MIDDLE_TIP, LM_MIDDLE_PIP),
        "ring":   finger_extended_from_wrist(lm, LM_RING_TIP,   LM_RING_PIP),
        "pinky":  finger_extended_from_wrist(lm, LM_PINKY_TIP,  LM_PINKY_PIP),
    }

def classify_hand_pose(fs: dict[str, bool]) -> GestureId:
    idx, mid, ring, pinky, thumb = fs["index"], fs["middle"], fs["ring"], fs["pinky"], fs["thumb"]
    n = sum([idx, mid, ring, pinky])
    if n == 0 and not thumb:                              return GestureId.FIST
    if n == 1 and idx:                                    return GestureId.INDEX_ONLY
    if n == 2 and idx and mid and not ring and not pinky: return GestureId.TWO_FINGERS
    if n == 3 and idx and mid and ring and not pinky:     return GestureId.THREE_FINGERS
    if n == 4 and idx and mid and ring and pinky:         return GestureId.FOUR_FINGERS
    if thumb and idx and mid and ring and pinky:          return GestureId.OPEN_PALM
    return GestureId.OTHER


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class HandSnapshot:
    label:        str
    lm:           list
    px:           list[tuple[int, int]]
    fs:           dict[str, bool]
    d_thumb_index:  float
    d_thumb_middle: float
    midpoint_im:  tuple[int, int]
    gesture:      GestureId

def build_hand_snapshot(label: str, lm, w: int, h: int) -> HandSnapshot:
    px    = [lm_to_px(lm[i], w, h) for i in range(21)]
    fs    = finger_states(lm)
    d_ti  = dist_px(px[LM_THUMB_TIP], px[LM_INDEX_TIP])
    d_tm  = dist_px(px[LM_THUMB_TIP], px[LM_MIDDLE_TIP])
    mid_x = (px[LM_INDEX_TIP][0] + px[LM_MIDDLE_TIP][0]) // 2
    mid_y = (px[LM_INDEX_TIP][1] + px[LM_MIDDLE_TIP][1]) // 2
    return HandSnapshot(
        label=label, lm=lm, px=px, fs=fs,
        d_thumb_index=d_ti, d_thumb_middle=d_tm,
        midpoint_im=(mid_x, mid_y),
        gesture=classify_hand_pose(fs),
    )


@dataclass
class HistoryEntry:
    label: str
    timestamp: float
    color: tuple[int, int, int]


@dataclass
class ControlState:
    """State for zero-collision 10-function gesture system."""
    last_fire_time:     dict[str, float]    = field(default_factory=dict)
    prev_two_finger_mid: tuple[int, int] | None = None
    tab_swipe_acc:      float = 0.0
    system_paused:      bool  = False
    fist_fired:         bool  = False

    # Gesture stack (rich history with timestamps for the HUD panel)
    history_stack: deque[HistoryEntry] = field(
        default_factory=lambda: deque(maxlen=CONFIG.HISTORY_SIZE)
    )
    # Simple string deque for window title
    title_history: deque[str] = field(
        default_factory=lambda: deque(maxlen=6)
    )

    def cooldown_ok(self, key: str, now: float, cooldown: float) -> bool:
        if now - self.last_fire_time.get(key, 0.0) >= cooldown:
            self.last_fire_time[key] = now
            return True
        return False

    def add_event(self, label: str, now: float) -> None:
        color = GESTURE_COLORS.get(label, CLR_WHITE)
        # Avoid duplicate consecutive entries in the stack
        if not self.history_stack or self.history_stack[-1].label != label:
            self.history_stack.append(HistoryEntry(label=label, timestamp=now, color=color))
        if not self.title_history or self.title_history[-1] != label:
            self.title_history.append(label)

    def get_title_status(self) -> str:
        return " | ".join(list(self.title_history)) if self.title_history else ""


# ─────────────────────────────────────────────────────────────────────────────
# Drawing primitives  (neon futuristic style)
# ─────────────────────────────────────────────────────────────────────────────
HAND_CONNECTIONS = mp_hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS


def draw_hand_skeleton(frame: np.ndarray, lm_list: list, now: float) -> None:
    """Renders hand bones as a neon cyan skeleton with glowing joints."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]

    # Bones
    for conn in HAND_CONNECTIONS:
        p1, p2 = pts[conn.start], pts[conn.end]
        # Outer glow
        cv2.line(frame, p1, p2, (120, 80, 0), 5, cv2.LINE_AA)
        # Inner bright line
        cv2.line(frame, p1, p2, CLR_CYAN, 2, cv2.LINE_AA)

    # Joints
    for i, p in enumerate(pts):
        r = 6 if i in (LM_INDEX_TIP, LM_MIDDLE_TIP, LM_RING_TIP, LM_PINKY_TIP, LM_THUMB_TIP) else 3
        cv2.circle(frame, p, r + 3, (80, 60, 0), -1)   # outer glow
        cv2.circle(frame, p, r, CLR_CYAN, -1)
        cv2.circle(frame, p, r - 1, (255, 255, 255), 1)  # spark core


def draw_neon_glow(
    frame: np.ndarray,
    point: tuple[int, int],
    color: tuple[int, int, int],
    radius: int,
    rings: int = 5,
) -> None:
    """Multi-ring alpha-blended neon glow at a point."""
    overlay = frame.copy()
    for i, r in enumerate(range(radius, radius + rings * 4, 4)):
        alpha = 0.55 * (1 - i / rings)
        cv2.circle(overlay, point, r, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        overlay = frame.copy()
    # Bright white core
    cv2.circle(frame, point, max(3, radius // 3), (255, 255, 255), -1)


def draw_neon_line(
    frame: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Glowing line — outer dark halo + bright inner."""
    cv2.line(frame, p1, p2, tuple(c // 4 for c in color), thickness + 6, cv2.LINE_AA)
    cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def draw_text_neon(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int] = CLR_CYAN,
    scale: float = 0.6,
    thickness: int = 1,
    bg_alpha: float = 0.55,
) -> None:
    """Text with a dark semi-transparent backing and a subtle neon shadow."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    pad = 5
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + pad), (10, 10, 10), -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    # Shadow
    cv2.putText(frame, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_scanlines(frame: np.ndarray, alpha: float = 0.06) -> None:
    """Subtle CRT scanline overlay for futuristic aesthetics."""
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame)
    overlay[::2, :] = 20   # dim every other row
    cv2.addWeighted(overlay, alpha, frame, 1.0, 0, frame)


def draw_animated_border(frame: np.ndarray, now: float, color: tuple) -> None:
    """Animated corner brackets that pulse."""
    h, w = frame.shape[:2]
    t = (math.sin(now * 3.0) + 1) / 2   # 0..1 pulsing
    bright = tuple(int(c * (0.6 + 0.4 * t)) for c in color)
    L = 30   # bracket arm length
    thick = 2
    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    dirs    = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    for (cx, cy), (dx, dy) in zip(corners, dirs):
        cv2.line(frame, (cx, cy), (cx + dx * L, cy), bright, thick)
        cv2.line(frame, (cx, cy), (cx, cy + dy * L), bright, thick)


def draw_header_bar(frame: np.ndarray, detected_hands: list[str], now: float) -> None:
    """Neon header bar with HAND presence indicators and a pulsing accent line."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (12, 12, 18), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # Pulsing accent line
    pulse = int(180 + 75 * math.sin(now * 4.0))
    cv2.line(frame, (0, 52), (w, 52), (pulse, pulse // 2, 0), 1)

    # Hand presence dots
    right_on = "Right" in detected_hands
    left_on  = "Left"  in detected_hands
    for label, xoff, active in [("LEFT", w - 180, left_on), ("RIGHT", w - 90, right_on)]:
        col = CLR_GREEN if active else (60, 60, 60)
        cx = xoff
        cv2.circle(frame, (cx - 10, 28), 6, col, -1)
        if active:
            cv2.circle(frame, (cx - 10, 28), 9, tuple(c // 3 for c in col), 1)
        draw_text_neon(frame, label, (cx, 36), color=col, scale=0.38)


def draw_gesture_stack(frame: np.ndarray, state: ControlState, now: float) -> None:
    """
    Right-side gesture history stack panel.
    Most recent gesture is at the top; older entries fade out.
    """
    h, w = frame.shape[:2]
    panel_w = CONFIG.HISTORY_PANEL_W
    panel_x = w - panel_w

    # Translucent panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, 56), (w, h), (8, 8, 14), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Panel border
    cv2.line(frame, (panel_x, 56), (panel_x, h), (40, 40, 60), 1)

    # Header
    draw_text_neon(frame, "GESTURE LOG", (panel_x + 10, 80), color=CLR_CYAN, scale=0.42)
    cv2.line(frame, (panel_x + 5, 88), (w - 5, 88), (40, 40, 60), 1)

    # Entries (newest at top)
    entries = list(reversed(state.history_stack))
    y = 110
    for entry in entries:
        age = now - entry.timestamp
        if age > CONFIG.HISTORY_FADE_S:
            continue
        # Fade alpha based on age
        fade = max(0.15, 1.0 - age / CONFIG.HISTORY_FADE_S)
        col = tuple(int(c * fade) for c in entry.color)

        # Time-ago string
        if age < 1.0:
            time_str = "now"
        elif age < 60:
            time_str = f"{age:.0f}s"
        else:
            time_str = f"{age/60:.1f}m"

        # Bullet dot
        cv2.circle(frame, (panel_x + 14, y - 5), 4, col, -1)

        draw_text_neon(frame, entry.label, (panel_x + 26, y), color=col, scale=0.42)
        draw_text_neon(frame, time_str,    (w - 38, y), color=tuple(c // 2 for c in col), scale=0.30)

        y += 24
        if y > h - 20:
            break


def draw_paused_overlay(frame: np.ndarray, now: float) -> None:
    """Full-frame dim + blinking PAUSED text."""
    overlay = np.zeros_like(frame)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    pulse = int(200 + 55 * math.sin(now * 5.0))
    color = (0, 0, pulse)
    h, w = frame.shape[:2]
    draw_text_neon(frame, "[ SYSTEM PAUSED ]", (w // 2 - 130, h // 2),
                   color=color, scale=1.0, thickness=2)
    draw_text_neon(frame, "Make a fist again to resume",
                   (w // 2 - 145, h // 2 + 40), color=(120, 120, 120), scale=0.45)


# ─────────────────────────────────────────────────────────────────────────────
# Gesture dispatcher
# ─────────────────────────────────────────────────────────────────────────────
def handle_controls(
    frame: np.ndarray,
    hands: list[HandSnapshot],
    state: ControlState,
    now:   float,
    screen_w: int,
    screen_h: int,
) -> None:
    """Zero-collision 10-function dispatcher. Mutates frame and state."""
    # Panel width to keep gestures inside the non-panel area
    fw = frame.shape[1]

    if not hands:
        state.prev_two_finger_mid = None
        state.tab_swipe_acc       = 0.0
        state.fist_fired          = False
        return

    hs = hands[0]
    px = hs.px
    lm = hs.lm
    g  = hs.gesture

    # ── 10: PAUSE / RESUME ──────────────────────────────────────────────────
    if g == GestureId.FIST:
        if not state.fist_fired:
            state.system_paused = not state.system_paused
            state.fist_fired    = True
            label = "PAUSED" if state.system_paused else "RESUMED"
            state.add_event(label, now)
            draw_neon_glow(frame, px[LM_WRIST], CLR_RED if state.system_paused else CLR_GREEN, 30)
        return
    else:
        state.fist_fired = False

    if state.system_paused:
        return   # render handled in main loop

    # ── 9: OPEN NEW TAB (open palm) ─────────────────────────────────────────
    if g == GestureId.OPEN_PALM:
        draw_neon_glow(frame, px[LM_WRIST], CLR_MAGENTA, 24)
        if state.cooldown_ok("new_tab", now, CONFIG.NEW_TAB_COOLDOWN):
            pyautogui.hotkey("ctrl", "t")
            state.add_event("New Tab", now)
        state.prev_two_finger_mid = None
        return

    # ── 8: VOLUME DOWN (4 fingers) ───────────────────────────────────────────
    if g == GestureId.FOUR_FINGERS:
        draw_neon_glow(frame, px[LM_MIDDLE_TIP], CLR_ORANGE, 18)
        if state.cooldown_ok("vol_down", now, CONFIG.VOLUME_COOLDOWN):
            pyautogui.press("volumedown")
            state.add_event("Vol DOWN", now)
        state.prev_two_finger_mid = None
        return

    # ── 7: VOLUME UP (3 fingers) ─────────────────────────────────────────────
    if g == GestureId.THREE_FINGERS:
        draw_neon_glow(frame, px[LM_MIDDLE_TIP], CLR_GREEN, 18)
        if state.cooldown_ok("vol_up", now, CONFIG.VOLUME_COOLDOWN):
            pyautogui.press("volumeup")
            state.add_event("Vol UP", now)
        state.prev_two_finger_mid = None
        return

    # ── 5 & 6: SCROLL + SWITCH TAB (2 fingers) ──────────────────────────────
    if g == GestureId.TWO_FINGERS:
        draw_neon_line(frame, px[LM_INDEX_TIP], px[LM_MIDDLE_TIP], CLR_CYAN)
        mid  = hs.midpoint_im
        prev = state.prev_two_finger_mid
        state.prev_two_finger_mid = mid

        if prev is not None:
            dy = mid[1] - prev[1]
            dx = mid[0] - prev[0]

            if abs(dy) > abs(dx):    # ── Vertical → Scroll
                state.tab_swipe_acc = 0.0
                if dy < -CONFIG.SCROLL_DELTA_Y and state.cooldown_ok("scroll_up", now, CONFIG.SCROLL_COOLDOWN):
                    pyautogui.scroll(3)
                    state.add_event("Scroll UP", now)
                    draw_neon_glow(frame, mid, CLR_BLUE, 16)
                    # Arrow indicator
                    cv2.arrowedLine(frame, (mid[0], mid[1]+30), (mid[0], mid[1]-30),
                                    CLR_BLUE, 2, tipLength=0.4)
                elif dy > CONFIG.SCROLL_DELTA_Y and state.cooldown_ok("scroll_dn", now, CONFIG.SCROLL_COOLDOWN):
                    pyautogui.scroll(-3)
                    state.add_event("Scroll DOWN", now)
                    draw_neon_glow(frame, mid, CLR_ORANGE, 16)
                    cv2.arrowedLine(frame, (mid[0], mid[1]-30), (mid[0], mid[1]+30),
                                    CLR_ORANGE, 2, tipLength=0.4)
            else:                    # ── Horizontal → Tab switch
                state.tab_swipe_acc += dx
                if state.tab_swipe_acc > CONFIG.SWITCH_TAB_ACCUM:
                    if state.cooldown_ok("tab_next", now, CONFIG.TAB_COOLDOWN):
                        pyautogui.hotkey("ctrl", "tab")
                        state.add_event("Next Tab", now)
                        draw_neon_glow(frame, mid, CLR_YELLOW, 16)
                    state.tab_swipe_acc = 0.0
                elif state.tab_swipe_acc < -CONFIG.SWITCH_TAB_ACCUM:
                    if state.cooldown_ok("tab_prev", now, CONFIG.TAB_COOLDOWN):
                        pyautogui.hotkey("ctrl", "shift", "tab")
                        state.add_event("Prev Tab", now)
                        draw_neon_glow(frame, mid, CLR_YELLOW, 16)
                    state.tab_swipe_acc = 0.0
        return
    else:
        state.prev_two_finger_mid = None
        state.tab_swipe_acc       = 0.0

    # ── 1 & 2: CURSOR MOVE / LEFT CLICK (index only) ────────────────────────
    if g == GestureId.INDEX_ONLY:
        if hs.d_thumb_index < CONFIG.INDEX_THUMB_TOUCH:
            # Left Click
            draw_neon_line(frame, px[LM_THUMB_TIP], px[LM_INDEX_TIP], CLR_GREEN)
            draw_neon_glow(frame, px[LM_INDEX_TIP], CLR_GREEN, 22)
            if state.cooldown_ok("left_click", now, CONFIG.CLICK_COOLDOWN):
                pyautogui.click()
                state.add_event("Left Click", now)
            return

        # Cursor Move — scale to non-panel area
        t8 = lm[LM_INDEX_TIP]
        sx = int(t8.x * screen_w)
        sy = int(t8.y * screen_h)
        pyautogui.moveTo(sx, sy, duration=CONFIG.CURSOR_SMOOTH_DURATION)
        # Crosshair cursor indicator
        cx, cy = px[LM_INDEX_TIP]
        arm = 12
        cv2.line(frame, (cx - arm, cy), (cx + arm, cy), CLR_YELLOW, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - arm), (cx, cy + arm), CLR_YELLOW, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 5, CLR_YELLOW, -1)
        draw_neon_glow(frame, (cx, cy), CLR_YELLOW, 8)
        return

    # ── 3: RIGHT CLICK (middle + thumb) ─────────────────────────────────────
    if hs.d_thumb_middle < CONFIG.MIDDLE_THUMB_TOUCH:
        draw_neon_line(frame, px[LM_THUMB_TIP], px[LM_MIDDLE_TIP], CLR_CYAN)
        draw_neon_glow(frame, px[LM_MIDDLE_TIP], CLR_CYAN, 22)
        if state.cooldown_ok("right_click", now, CONFIG.CLICK_COOLDOWN):
            pyautogui.rightClick()
            state.add_event("Right Click", now)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
WIN = "Hand Gesture Control"

def main() -> None:
    pyautogui.FAILSAFE = False  # prevent accidental failsafe corner-quit

    screen_w, screen_h = pyautogui.size()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS,           30)

    model_path   = ensure_hand_landmarker_model()
    hand_options = mp_hand_landmarker.HandLandmarkerOptions(
        base_options=mp_base_options.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision_mode.VisionTaskRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.70,
        min_hand_presence_confidence=0.50,
        min_tracking_confidence=0.50,
    )

    frame_ts_ms   = 0
    control_state = ControlState()

    # ── Create window — always on top so gestures never close it ────────────
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)   # ★ stays on top

    with mp_hand_landmarker.HandLandmarker.create_from_options(hand_options) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue

            frame      = cv2.flip(frame, 1)
            fh, fw     = frame.shape[:2]
            now        = time.monotonic()
            frame_ts_ms += 33

            # ── MediaPipe inference ─────────────────────────────────────────
            rgb    = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_img = MpImage(image_format=ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, frame_ts_ms)

            hands: list[HandSnapshot] = []
            if result.hand_landmarks and result.handedness:
                for lm_list, handedness_list in zip(result.hand_landmarks, result.handedness):
                    label = handedness_list[0].category_name if handedness_list else "Unknown"
                    hands.append(build_hand_snapshot(label, lm_list, fw, fh))

            # ── Gesture logic ───────────────────────────────────────────────
            handle_controls(frame, hands, control_state, now, screen_w, screen_h)

            # ── Draw skeletons ──────────────────────────────────────────────
            if result.hand_landmarks:
                for lm_list in result.hand_landmarks:
                    draw_hand_skeleton(frame, lm_list, now)

            # ── Futuristic UI overlays ──────────────────────────────────────
            draw_scanlines(frame, alpha=0.08)
            current_labels = [h.label for h in hands]
            draw_header_bar(frame, current_labels, now)
            draw_gesture_stack(frame, control_state, now)
            draw_animated_border(frame, now,
                                  CLR_RED if control_state.system_paused else CLR_CYAN)

            if control_state.system_paused:
                draw_paused_overlay(frame, now)

            # ── Window title ────────────────────────────────────────────────
            title = "Hand Gesture Control"
            status = control_state.get_title_status()
            cv2.setWindowTitle(WIN, f"{title}  |  {status}" if status else title)

            # ── Keep window on top every frame (robust on Windows) ──────────
            cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)

            cv2.imshow(WIN, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
