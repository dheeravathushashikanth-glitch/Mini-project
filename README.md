# Hand gesture virtual mouse

Control the mouse and desktop shortcuts from your webcam using **MediaPipe** hand landmarks and **PyAutoGUI**.  
Based on [HaticeSude/hand-gesture-virtual-mouse](https://github.com/HaticeSude/hand-gesture-virtual-mouse), extended with a **unified gesture engine** (`unified_hand_app.py`): distance-based shortcuts, pinch scrolling, two-finger scroll, fist drag, and smoother cursor control.

---

## Features

### Pointer and mouse (by hand)

| Hand   | Gesture / pose | Action |
|--------|----------------|--------|
| **Right** | Index extended (pointing) | Move cursor |
| **Right** | **V** (index + middle up, ring/pinky down) | Move cursor with smoothing (`duration=0.1`) |
| **Left** | Thumb + index close | Left click |
| **Left** | Thumb + index PIP close | Double click |
| **Left** | Thumb + pinky close | Right click |

### System and browser (either hand, with debounce)

| Gesture | Condition | Action |
|---------|-----------|--------|
| Volume down | Index–middle tip distance **40–100 px** | `volumedown` |
| Volume up | Index–middle distance **> 100 px** | `volumeup` |
| Next tab | Middle extended, thumb–middle **> 150 px** | `Ctrl+Tab` |
| New tab | Index + pinky extended, large span | `Ctrl+T` |
| Back / forward | Fast horizontal index motion | `Alt+Left` / `Alt+Right` |

### Advanced

- **Fist** — all fingers closed: hold **mouse drag** (release when hand opens).
- **Pinch** (thumb–index): vertical move → **scroll**; horizontal move → **horizontal scroll** (or Shift+scroll fallback).
- **Two-finger scroll** — index + middle up with tips **close together**: move the pair vertically to scroll.
- **Multi-hand** — up to two hands; state is tracked per **Left/Right** label.
- **Green lines** on the preview show which fingertip pairs are used for distance rules.
- **Edge-triggered actions** — shortcuts do not repeat while you hold the same pose (release to arm again).

---

## Tech stack

- **Python 3.10+** (3.11 recommended; MediaPipe Tasks wheels are unreliable on some 3.13 setups)
- **OpenCV** — camera and preview
- **MediaPipe Tasks** — `HandLandmarker` (legacy `mp.solutions` is not used in the main app)
- **PyAutoGUI** — mouse and keyboard
- **NumPy**

On first run the app downloads **`hand_landmarker.task`** (~10 MB) into **`models/`** (ignored by git).

---

## Project layout

```
hand-gesture-virtual-mouse/
├── src/
│   ├── unified_hand_app.py   # Full app: detection, gestures, main loop
│   ├── el_mouse_kontrol.py   # Entry shim → runs unified_hand_app.main()
│   ├── el_tespiti.py         # Legacy hand demo (uses old MediaPipe API)
│   └── kamera_test.py        # Webcam test
├── models/                   # Created at runtime; holds .task model
├── docs/                     # Gesture reference images
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Run (either command):**

```bash
python src/unified_hand_app.py
python src/el_mouse_kontrol.py
```

Allow **camera** access when prompted. Focus the OpenCV window and press **Q** to quit.

---

## Tips

- Use good lighting and keep your full hand in frame.
- **Mirror view** matches natural left/right; MediaPipe still labels hands as **Left/Right** from the camera’s perspective.
- Tune thresholds in `unified_hand_app.py` (e.g. `CLICK_THRESHOLD`, `PINCH_MAX_DIST`, swipe and volume bands) if gestures feel too sensitive or too stiff.

---

## Credits

- **Original project:** Hatice Sude Mutlu — [HaticeSude on GitHub](https://github.com/HaticeSude)
- **Extensions:** unified gesture suite, MediaPipe Tasks migration, extra shortcuts and scrolling — maintained in this fork.

---

## Possible future work

- Configuration UI (sensitivity, camera index, threshold presets)
- Optional stability filtering for jittery fingertip tracking
