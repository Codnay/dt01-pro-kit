"""Shared live camera preview helper for interactive examples."""

from __future__ import annotations

import os
import sys

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency at runtime
    cv2 = None


class CameraPreview:
    """Optional OpenCV preview window for RGB/depth frames."""

    def __init__(self, *, enabled: bool, window_name: str, rotation: int = 0) -> None:
        self._enabled = enabled
        self._window_name = window_name
        self._window_created = False
        self._warning_printed = False
        self._rotation = rotation % 360

        if self._enabled and cv2 is None:
            print("OpenCV not available. Camera preview disabled.")
            self._enabled = False
            return

        if self._enabled and not self._is_display_available():
            print("No GUI display detected. Camera preview disabled.")
            print("   - If running over SSH, use X11 forwarding (`ssh -X`) or run locally.")
            print("   - Or run without `--show-camera-preview`.")
            self._enabled = False
            return

        if self._enabled:
            highgui_ok, exc = self._probe_highgui()
            if not highgui_ok:
                print(f"OpenCV GUI backend unavailable. Camera preview disabled: {exc}")
                self._print_fix_instructions()
                self._enabled = False

    @staticmethod
    def _is_display_available() -> bool:
        """Return whether a GUI display is available for HighGUI windows."""
        if sys.platform.startswith("linux"):
            return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        # macOS/Windows generally have a display session when running locally.
        return True

    def _probe_highgui(self) -> tuple[bool, Exception | None]:
        """Try creating a tiny OpenCV window to verify HighGUI support."""
        if cv2 is None:
            return False, RuntimeError("OpenCV is not importable")

        probe_name = f"{self._window_name} [probe]"
        probe_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        try:
            cv2.namedWindow(probe_name, cv2.WINDOW_NORMAL)
            cv2.imshow(probe_name, probe_frame)
            cv2.waitKey(1)
            cv2.destroyWindow(probe_name)
            return True, None
        except Exception as exc:
            return False, exc

    @staticmethod
    def _print_fix_instructions() -> None:
        """Print practical remediation steps for missing OpenCV GUI support."""
        print("   Fix:")
        print(
            "   1) Remove headless OpenCV builds:"
            " python -m pip uninstall -y opencv-python-headless opencv-contrib-python-headless"
        )
        print("   2) Install GUI-enabled OpenCV: python -m pip install --upgrade opencv-python")
        if sys.platform.startswith("linux"):
            print("   3) On Ubuntu/Debian, install GUI libs: sudo apt-get install -y libgtk-3-dev pkg-config")
        print("   4) Restart the Python process and rerun with `--show-camera-preview`.")

    @property
    def enabled(self) -> bool:
        """Return whether preview updates are active."""
        return self._enabled

    @staticmethod
    def _render_depth_bgr(depth_image_mm: np.ndarray | None) -> np.ndarray | None:
        """Render depth (uint16 mm) into a human-readable BGR visualization."""
        if depth_image_mm is None or cv2 is None:
            return None

        depth = depth_image_mm.astype(np.float32)
        valid = depth > 0
        if not np.any(valid):
            normalized = np.zeros_like(depth, dtype=np.uint8)
        else:
            lo, hi = np.percentile(depth[valid], [2.0, 98.0])
            if hi <= lo:
                hi = lo + 1.0
            normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
            normalized = (normalized * 255.0).astype(np.uint8)

        return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)

    def update(
        self,
        rgb_image: np.ndarray | None,
        depth_image_mm: np.ndarray | None = None,
    ) -> None:
        """Render latest RGB/depth frame if preview is enabled."""
        if not self._enabled or cv2 is None:
            return

        try:
            rgb_bgr = (
                cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                if rgb_image is not None
                else None
            )
            depth_bgr = self._render_depth_bgr(depth_image_mm)

            if rgb_bgr is None and depth_bgr is None:
                return
            if rgb_bgr is None:
                preview = depth_bgr
            elif depth_bgr is None:
                preview = rgb_bgr
            else:
                if depth_bgr.shape[:2] != rgb_bgr.shape[:2]:
                    depth_bgr = cv2.resize(
                        depth_bgr, (rgb_bgr.shape[1], rgb_bgr.shape[0])
                    )
                preview = np.hstack([rgb_bgr, depth_bgr])

            if self._rotation:
                # np.rot90 rotates counterclockwise; negate k for clockwise degrees.
                k = (4 - self._rotation // 90) % 4
                preview = np.rot90(preview, k=k)

            cv2.imshow(self._window_name, preview)
            cv2.waitKey(1)
            self._window_created = True
        except Exception as exc:
            if not self._warning_printed:
                print(f"Camera preview disabled due to OpenCV error: {exc}")
                self._warning_printed = True
            self._enabled = False

    def close(self) -> None:
        """Close preview window if it was created."""
        if cv2 is None or not self._window_created:
            return

        try:
            cv2.destroyWindow(self._window_name)
        except Exception:
            pass
