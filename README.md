# 🖐️ Hand Gesture Virtual Mouse

> **Control your entire desktop — cursor, clicks, scroll, volume, tabs — with nothing but your hand and a webcam.**

Built with **MediaPipe HandLandmarker**, **OpenCV**, and **PyAutoGUI**. Features a zero-collision 10-function gesture engine with a premium neon-glow HUD, persistent gesture history, and a fully centralized configuration system.

---

## ✨ Features at a Glance

- 🖱️ **Cursor Control** — Point with one finger to move the mouse
- 👆 **Left Click** — Touch index tip to thumb
- 🤏 **Right Click** — Touch middle tip to thumb
- ✌️ **Scroll** — Two fingers moving up/down
- ↔️ **Switch Tabs** — Two fingers swiping left/right
- 🎵 **Volume Up/Down** — 3 or 4 fingers raised
- 🌐 **Open New Tab** — Open palm (all 5 fingers)
- ⏸️ **Pause/Resume System** — Closed fist toggles all gesture detection
- 🌟 **Neon Glow HUD** — Real-time visual feedback for every action
- 📜 **Gesture History** — Window title shows your last 5 actions

---

## 🎯 Zero-Collision Gesture Table

Each gesture is triggered by a **unique finger combination** — no two gestures share the same shape. Scroll vs Tab Switch is resolved by **axis dominance** (vertical vs horizontal motion).

| #  | Operation      | Gesture                                   | Fingers Used                     | Why It Won't Collide                              |
|----|----------------|-------------------------------------------|-----------------------------------|---------------------------------------------------|
| 1  | **Cursor Move** | ☝️ Index finger extended, others closed   | Index (8) only                   | Very unique single-finger shape                   |
| 2  | **Left Click**  | 👆 + 👍 Index + Thumb touch               | Index (8) + Thumb (4) touch      | Different from movement — requires touch distance |
| 3  | **Right Click** | 🖕 + 👍 Middle + Thumb touch              | Middle (12) + Thumb (4) touch    | Different fingers from left click                 |
| 4  | **Scroll Up**   | ✌️ Two fingers moving **UP**              | Index (8) + Middle (12) vertical | Unique two-finger pose + direction                |
| 5  | **Scroll Down** | ✌️ Two fingers moving **DOWN**            | Index (8) + Middle (12) vertical | Same pose, opposite direction                     |
| 6  | **Switch Tab →** | ✌️ Two fingers swiping **RIGHT**         | Index (8) + Middle (12) horizontal | Axis-split from scroll                          |
| 7  | **Switch Tab ←** | ✌️ Two fingers swiping **LEFT**          | Index (8) + Middle (12) horizontal | Axis-split from scroll                          |
| 8  | **Volume Up**   | 🤟 Three fingers up                       | Index + Middle + Ring            | Different finger count (3 vs 2 or 4)             |
| 9  | **Volume Down** | 🖐️ Four fingers up (no thumb)            | Index + Middle + Ring + Pinky    | Clearly different count                          |
| 10 | **Open New Tab** | 🖐️ Full open palm (all 5)               | All 5 fingers                    | Easy to detect, maximum spread                   |
| 🔒 | **Pause System** | ✊ Closed fist                           | 0 fingers — all closed           | Most distinct shape, zero conflict               |

---

## 🎨 Visual Feedback (HUD)

| Indicator              | Color           | Meaning                                      |
|------------------------|-----------------|----------------------------------------------|
| Yellow dot             | 🟡 Gold         | Index cursor tracking                        |
| Green line             | 🟢 Green        | Active gesture pair (click, scroll, etc.)    |
| Neon glow — teal/cyan  | 🔵 Cyan         | Open palm / New Tab triggered                |
| Neon glow — green      | 🟢 Bright green | Volume Up triggered                          |
| Neon glow — orange     | 🟠 Orange       | Volume Down triggered                        |
| Neon glow — green      | 🟢 Lime green   | Left click triggered                         |
| Neon glow — cyan       | 🔵 Teal         | Right click triggered                        |
| Scroll arrow — blue    | 🔵 Blue         | Scroll Up visual arrow                       |
| Scroll arrow — orange  | 🟠 Orange       | Scroll Down visual arrow                     |
| Neon glow — dark blue  | 🔵 Navy         | System Paused (fist detected)                |
| Dashboard bar          | ⬛ Dark overlay  | Hand presence indicator: LEFT / RIGHT (green if active) |
| Window title           | _text_          | Last 5 gestures shown as `VOL UP \| Left Click \| ...` |

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.10–3.12** (3.11 recommended)
- Webcam

### 2. Install

```powershell
# Clone the repository
git clone https://github.com/dheeravathushashikanth-glitch/Mini-project.git
cd Mini-project

# Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Run

```powershell
python src/unified_hand_app.py
# OR
python src/el_mouse_kontrol.py
```

On **first run**, the app downloads the MediaPipe `hand_landmarker.task` model (~10 MB) automatically into `models/`.

Press **Q** in the camera window to quit.

---

## 🏗️ Project Structure

```
hand-gesture-virtual-mouse/
├── src/
│   ├── unified_hand_app.py   # ★ Main app — all gesture logic, HUD, main loop
│   ├── el_mouse_kontrol.py   # Entry shim → calls unified_hand_app.main()
│   ├── kamera_test.py        # Webcam sanity test
│   └── el_tespiti.py         # Legacy hand detection demo
├── models/                   # Auto-created; holds downloaded .task model file
├── docs/                     # Gesture reference images
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

All thresholds and timings are centralized in the `AppConfig` class at the top of `unified_hand_app.py`. Edit these to tune sensitivity without touching gesture logic:

```python
@dataclass(frozen=True)
class AppConfig:
    # Touch thresholds (pixels)
    INDEX_THUMB_TOUCH: int = 38     # Left Click sensitivity
    MIDDLE_THUMB_TOUCH: int = 38    # Right Click sensitivity

    # Motion thresholds (pixels per frame)
    SCROLL_DELTA_Y: int = 12        # Scroll trigger distance
    SWITCH_TAB_ACCUM: int = 60      # Tab swipe accumulation

    # Cooldowns (seconds)
    CLICK_COOLDOWN: float = 0.4
    VOLUME_COOLDOWN: float = 0.35
    TAB_COOLDOWN: float = 0.5
    NEW_TAB_COOLDOWN: float = 1.0
    SCROLL_COOLDOWN: float = 0.08
```

---

## 🧠 Architecture

```
Camera Frame (OpenCV)
        │
        ▼
MediaPipe HandLandmarker
  (21 landmarks per hand)
        │
        ▼
build_hand_snapshot()
  ├── finger_states()       → Which fingers are extended
  ├── classify_hand_pose()  → Maps to GestureId enum
  ├── dist_px()             → Pixel distances (thumb-index, thumb-middle)
  └── midpoint_im           → Two-finger midpoint for scroll/tab tracking
        │
        ▼
handle_controls()
  ├── GestureId.FIST        → Toggle system pause
  ├── GestureId.OPEN_PALM   → Open new tab
  ├── GestureId.FOUR_FINGERS→ Volume down
  ├── GestureId.THREE_FINGERS→ Volume up
  ├── GestureId.TWO_FINGERS  → Scroll (vertical) / Switch Tab (horizontal)
  ├── GestureId.INDEX_ONLY  → Cursor move / Left click (if thumb touches)
  └── d_thumb_middle < threshold → Right click
        │
        ▼
draw_premium_dashboard()    → Top HUD bar with LEFT/RIGHT indicators
draw_neon_glow()            → Layered transparent circles for effects
draw_text_with_bg()         → Semi-transparent text overlay
        │
        ▼
cv2.imshow() + window title update (gesture history)
```

---

## 🔧 Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `mediapipe` | ≥ 0.10 | Hand landmark detection (Tasks API) |
| `opencv-python` | ≥ 4.8 | Camera capture + frame rendering |
| `pyautogui` | ≥ 0.9 | Mouse + keyboard control |
| `numpy` | ≥ 1.24 | Frame array manipulation |

---

## 💡 Tips for Best Performance

1. **Good lighting** — Ensure your hand is evenly lit; avoid strong backlight.
2. **Keep hand in frame** — All 21 landmarks must be visible for accurate detection.
3. **Steady transitions** — Move deliberately between gestures; rapid switching may cause brief overlap.
4. **Tune thresholds** — If clicks fire accidentally, increase `INDEX_THUMB_TOUCH`. If they don't register, decrease it.
5. **System Pause** — Use the **closed fist** to freeze all gestures when you need to move your hand without triggering actions.

---

## 📋 Requirements

```
opencv-python
mediapipe
pyautogui
numpy
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

- Original project: [HaticeSude/hand-gesture-virtual-mouse](https://github.com/HaticeSude/hand-gesture-virtual-mouse)
- [Google MediaPipe](https://developers.google.com/mediapipe) for the HandLandmarker model
- Built and extended with ❤️ using [Antigravity AI](https://antigravity.dev)
