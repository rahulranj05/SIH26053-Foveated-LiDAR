"""
A39.2 — FoveaMap Visual Dashboard

A lightweight visual presentation layer built on top of the A39.1
DashboardFrame data adapter.

This module does not modify or execute the FoveaMap mapping pipeline.
It renders already-computed DashboardFrame objects using matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mapping.foveamap_dashboard import DashboardFrame


SAFETY_MODE_TITLES = {
    "NORMAL": "NORMAL",
    "DEGRADED": "DEGRADED",
    "SAFETY": "SAFETY",
}


@dataclass(frozen=True)
class VisualDashboardConfig:
    """Configuration for the A39.2 visual dashboard."""

    point_size: float = 2.0
    history_size: int = 100
    elevation_min: float | None = None
    elevation_max: float | None = None

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.point_size)
            or self.point_size <= 0.0
        ):
            raise ValueError(
                "point_size must be finite and > 0"
            )

        if (
            not isinstance(self.history_size, int)
            or self.history_size < 1
        ):
            raise ValueError(
                "history_size must be an integer >= 1"
            )

        if (
            self.elevation_min is not None
            and not np.isfinite(self.elevation_min)
        ):
            raise ValueError(
                "elevation_min must be finite or None"
            )

        if (
            self.elevation_max is not None
            and not np.isfinite(self.elevation_max)
        ):
            raise ValueError(
                "elevation_max must be finite or None"
            )

        if (
            self.elevation_min is not None
            and self.elevation_max is not None
            and self.elevation_min >= self.elevation_max
        ):
            raise ValueError(
                "elevation_min must be < elevation_max"
            )


@dataclass(frozen=True)
class VisualDashboardSnapshot:
    """Immutable summary of the current visual dashboard state."""

    frame_id: Any
    safety_mode: str
    point_count: int
    leaf_count: int
    latency_ms: float
    fps: float


def _validate_frame(
    frame: DashboardFrame,
) -> DashboardFrame:
    if not isinstance(
        frame,
        DashboardFrame,
    ):
        raise TypeError(
            "frame must be a DashboardFrame"
        )

    return frame


def _safe_fps_text(
    fps: float,
) -> str:
    if np.isinf(fps):
        return "∞"

    return f"{fps:.2f}"


def _mode_label(
    mode: str,
) -> str:
    return SAFETY_MODE_TITLES.get(
        mode,
        str(mode),
    )


class FoveaMapVisualDashboard:
    """
    Interactive matplotlib dashboard for DashboardFrame objects.

    The dashboard maintains only presentation history. It does not
    modify the supplied frame or underlying mapping results.
    """

    def __init__(
        self,
        *,
        config: VisualDashboardConfig | None = None,
    ) -> None:
        if config is None:
            config = VisualDashboardConfig()

        self.config = config

        self._frames: list[DashboardFrame] = []

        self._figure = None
        self._axes: dict[str, Any] = {}

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def latest_frame(
        self,
    ) -> DashboardFrame | None:
        if not self._frames:
            return None

        return self._frames[-1]

    def snapshot(
        self,
    ) -> VisualDashboardSnapshot | None:
        frame = self.latest_frame

        if frame is None:
            return None

        return VisualDashboardSnapshot(
            frame_id=frame.frame_id,
            safety_mode=frame.safety_mode,
            point_count=frame.point_count,
            leaf_count=frame.leaf_count,
            latency_ms=frame.processing_latency_ms,
            fps=frame.fps,
        )

    def update(
        self,
        frame: DashboardFrame,
    ) -> VisualDashboardSnapshot:
        """
        Add one DashboardFrame to the visual history.

        The frame is retained as immutable presentation input.
        """

        validated = _validate_frame(
            frame
        )

        self._frames.append(
            validated
        )

        if len(self._frames) > self.config.history_size:
            self._frames = self._frames[
                -self.config.history_size:
            ]

        return VisualDashboardSnapshot(
            frame_id=validated.frame_id,
            safety_mode=validated.safety_mode,
            point_count=validated.point_count,
            leaf_count=validated.leaf_count,
            latency_ms=validated.processing_latency_ms,
            fps=validated.fps,
        )

    def clear(
        self,
    ) -> None:
        """Clear presentation history."""

        self._frames.clear()

    def _create_figure(
        self,
    ) -> None:
        import matplotlib.pyplot as plt

        if self._figure is not None:
            return

        figure = plt.figure(
            figsize=(16, 10),
        )

        grid = figure.add_gridspec(
            3,
            4,
        )

        axes = {
            "point_cloud": figure.add_subplot(
                grid[:, :2]
            ),
            "status": figure.add_subplot(
                grid[0, 2:]
            ),
            "latency": figure.add_subplot(
                grid[1, 2]
            ),
            "fps": figure.add_subplot(
                grid[1, 3]
            ),
            "resolution": figure.add_subplot(
                grid[2, 2]
            ),
            "reasons": figure.add_subplot(
                grid[2, 3]
            ),
        }

        figure.suptitle(
            "FOVEAMAP — A39.2 VISUAL DASHBOARD",
            fontsize=16,
            fontweight="bold",
        )

        self._figure = figure
        self._axes = axes

    def _render_point_cloud(
        self,
        frame: DashboardFrame,
    ) -> None:
        axis = self._axes[
            "point_cloud"
        ]

        axis.clear()

        points = frame.preview_xyz

        axis.set_title(
            "2.5D LiDAR Preview"
        )

        axis.set_xlabel(
            "X (m)"
        )

        axis.set_ylabel(
            "Y (m)"
        )

        axis.set_aspect(
            "equal",
            adjustable="box",
        )

        if points.shape[0] == 0:
            axis.text(
                0.5,
                0.5,
                "No preview points",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axis.transAxes,
            )
            return

        z = points[:, 2]

        scatter_kwargs: dict[str, Any] = {
            "s": self.config.point_size,
            "c": z,
            "cmap": "viridis",
        }

        if self.config.elevation_min is not None:
            scatter_kwargs["vmin"] = (
                self.config.elevation_min
            )

        if self.config.elevation_max is not None:
            scatter_kwargs["vmax"] = (
                self.config.elevation_max
            )

        axis.scatter(
            points[:, 0],
            points[:, 1],
            **scatter_kwargs,
        )

        axis.grid(
            True,
            alpha=0.3,
        )

    def _render_status(
        self,
        frame: DashboardFrame,
    ) -> None:
        axis = self._axes[
            "status"
        ]

        axis.clear()
        axis.axis("off")

        reasons = (
            ", ".join(frame.safety_reasons)
            if frame.safety_reasons
            else "NONE"
        )

        text = "\n".join(
            [
                f"MODE: {_mode_label(frame.safety_mode)}",
                f"FRAME: {frame.frame_id}",
                f"POINTS: {frame.point_count:,}",
                (
                    "VALID: "
                    f"{frame.valid_point_count:,} "
                    f"({frame.valid_fraction:.1%})"
                ),
                (
                    "INVALID: "
                    f"{frame.invalid_point_count:,} "
                    f"({frame.invalid_fraction:.1%})"
                ),
                f"LEAVES: {frame.leaf_count:,}",
                (
                    "SEMANTIC: "
                    f"{'YES' if frame.semantic_available else 'NO'}"
                ),
                (
                    "DYNAMIC: "
                    f"{'YES' if frame.dynamic_available else 'NO'}"
                ),
                f"REASONS: {reasons}",
            ]
        )

        axis.text(
            0.02,
            0.95,
            text,
            transform=axis.transAxes,
            verticalalignment="top",
            fontsize=11,
            family="monospace",
        )

        axis.set_title(
            "System Status"
        )

    def _history_values(
        self,
        attribute: str,
    ) -> np.ndarray:
        return np.asarray(
            [
                getattr(
                    frame,
                    attribute,
                )
                for frame in self._frames
            ],
            dtype=np.float64,
        )

    def _render_latency(
        self,
    ) -> None:
        axis = self._axes[
            "latency"
        ]

        axis.clear()

        values = self._history_values(
            "processing_latency_ms"
        )

        axis.plot(
            values,
            marker="o",
            markersize=3,
        )

        axis.set_title(
            "Latency (ms)"
        )

        axis.set_xlabel(
            "Dashboard Frame"
        )

        axis.grid(
            True,
            alpha=0.3,
        )

    def _render_fps(
        self,
    ) -> None:
        axis = self._axes[
            "fps"
        ]

        axis.clear()

        values = self._history_values(
            "fps"
        )

        finite_values = values[
            np.isfinite(values)
        ]

        if finite_values.size > 0:
            axis.plot(
                finite_values,
                marker="o",
                markersize=3,
            )

        axis.set_title(
            "FPS"
        )

        axis.set_xlabel(
            "Dashboard Frame"
        )

        axis.grid(
            True,
            alpha=0.3,
        )

    def _render_resolution(
        self,
        frame: DashboardFrame,
    ) -> None:
        axis = self._axes[
            "resolution"
        ]

        axis.clear()

        counts = frame.resolution_counts

        axis.set_title(
            "Resolution Distribution"
        )

        if not counts:
            axis.text(
                0.5,
                0.5,
                "No resolution data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axis.transAxes,
            )
            return

        resolutions = [
            float(value)
            for value in counts.keys()
        ]

        values = [
            int(value)
            for value in counts.values()
        ]

        labels = [
            f"{resolution:.2f} m"
            for resolution in resolutions
        ]

        axis.bar(
            labels,
            values,
        )

        axis.tick_params(
            axis="x",
            rotation=45,
        )

    def _render_reasons(
        self,
        frame: DashboardFrame,
    ) -> None:
        axis = self._axes[
            "reasons"
        ]

        axis.clear()

        counts = (
            frame.dominant_reason_counts
        )

        axis.set_title(
            "Foveation Reasons"
        )

        if not counts:
            axis.text(
                0.5,
                0.5,
                "No foveation reason data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axis.transAxes,
            )
            return

        labels = list(
            counts.keys()
        )

        values = list(
            counts.values()
        )

        axis.bar(
            labels,
            values,
        )

        axis.tick_params(
            axis="x",
            rotation=45,
        )

    def render(
        self,
        frame: DashboardFrame | None = None,
    ):
        """
        Render the latest dashboard state.

        If frame is supplied, it is first added to dashboard history.
        """

        if frame is not None:
            self.update(frame)

        current = self.latest_frame

        if current is None:
            raise RuntimeError(
                "cannot render dashboard without a frame"
            )

        self._create_figure()

        self._render_point_cloud(
            current
        )

        self._render_status(
            current
        )

        self._render_latency()

        self._render_fps()

        self._render_resolution(
            current
        )

        self._render_reasons(
            current
        )

        self._figure.tight_layout(
            rect=(0.0, 0.0, 1.0, 0.96)
        )

        return self._figure

    def show(
        self,
        frame: DashboardFrame | None = None,
        *,
        block: bool = True,
    ) -> None:
        """Render and display the dashboard."""

        figure = self.render(
            frame
        )

        import matplotlib.pyplot as plt

        figure.canvas.draw_idle()

        plt.show(
            block=block
        )

    def save(
        self,
        path: str,
        frame: DashboardFrame | None = None,
        *,
        dpi: int = 150,
    ) -> str:
        """Render and save the dashboard image."""

        if (
            not isinstance(dpi, int)
            or dpi < 1
        ):
            raise ValueError(
                "dpi must be an integer >= 1"
            )

        figure = self.render(
            frame
        )

        figure.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
        )

        return path


def create_visual_dashboard(
    *,
    config: VisualDashboardConfig | None = None,
) -> FoveaMapVisualDashboard:
    """Create an A39.2 visual dashboard."""

    return FoveaMapVisualDashboard(
        config=config
    )