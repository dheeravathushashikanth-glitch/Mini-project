"""
Backward-compatible entry point for the unified hand-gesture application.

All logic lives in ``unified_hand_app`` (single project). Run either:

  python src/el_mouse_kontrol.py
  python src/unified_hand_app.py
"""

from unified_hand_app import main

if __name__ == "__main__":
    main()
