"""Calibration gate: held-out anchor accuracy, exits non-zero when the median exceeds the gate.

    uv run python scripts/gate_calibration.py --config <clip>.yaml \\
        --calibration outputs/<clip>/calibration.npz \\
        --anchors data/annotations/pitch_points_<clip>.json --gate-median-m 0.5
"""

from football_tracker.homography.gate import main

if __name__ == "__main__":
    main()
