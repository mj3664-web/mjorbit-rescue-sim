"""Browser-based manual action console with live mission sensor values."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import viser
from viewer.bodies import MuJoCoScene

from rescue_sim.simulator import (
    RescueConfig,
    RescueSimulator,
    sensor_packet_to_serializable,
)


@dataclass(frozen=True)
class ManualViewerConfig:
    """Network and timing configuration for the manual control console."""

    host: str = "0.0.0.0"
    port: int = 8080
    auto_run: bool = False
    duration_s: float | None = None


def sensor_summary_markdown(
    sensors: Mapping[str, Any], action: np.ndarray, reward: float | None
) -> str:
    """Format the high-value navigation sensors for the viewer dashboard."""

    relative_speed = float(
        np.linalg.norm(np.asarray(sensors["relative_velocity_world_m_s"]))
    )
    reward_text = "—" if reward is None else f"{reward:.3f}"
    action_text = np.array2string(action, precision=2, suppress_small=True)
    return (
        f"**Mission phase:** `{sensors['mission_phase']}`  \n"
        f"**Simulation time:** `{float(sensors['time_s']):.2f} s`  \n"
        f"**Probe → handle:** `{float(sensors['probe_handle_range_m']):.3f} m`  \n"
        f"**Relative speed:** `{relative_speed:.3f} m/s`  \n"
        f"**Distance to spacecraft:** "
        f"`{float(sensors['distance_to_spacecraft_m']):.3f} m`  \n"
        f"**Capture attempts:** `{int(sensors['capture_attempts'])}`  \n"
        f"**Last capture result:** `{sensors['last_capture_reason']}`  \n"
        f"**Last reward:** `{reward_text}`  \n"
        f"**Applied action:** `{action_text}`"
    )


class ManualRescueApp:
    """Interactive action sliders and sensor telemetry around the simulator."""

    def __init__(
        self,
        simulator: RescueSimulator,
        config: ManualViewerConfig | None = None,
    ) -> None:
        self.simulator = simulator
        self.config = config or ManualViewerConfig()
        self.server = viser.ViserServer(
            host=self.config.host,
            port=self.config.port,
            label="mjorbit astronaut rescue",
        )
        self.server.scene.set_up_direction("+z")
        self.server.initial_camera.position = (14.0, -16.0, 10.0)
        self.server.initial_camera.look_at = (5.5, 1.0, 0.5)

        model, _ = self.simulator.visualization_handles
        self.scene = MuJoCoScene(self.server, model, root_path="/mission/bodies")
        self.action = np.zeros(self.simulator.ACTION_SIZE, dtype=np.float64)
        self.action[6] = -1.0
        self._running = self.config.auto_run
        self._step_requested = False
        self._reset_requested = False
        self._capture_requested = False
        self._last_reward: float | None = None
        self._last_info: Mapping[str, Any] = {}
        self._playback_speed = 1.0

        self._probe = self.server.scene.add_icosphere(
            "/mission/markers/capture_probe",
            radius=0.09,
            color=(255, 210, 35),
        )
        self._handle = self.server.scene.add_icosphere(
            "/mission/markers/rescue_handle",
            radius=0.09,
            color=(255, 75, 45),
        )
        self._port = self.server.scene.add_icosphere(
            "/mission/markers/spacecraft_port",
            radius=0.12,
            color=(40, 210, 255),
            wireframe=True,
        )
        self._capture_line = self.server.scene.add_line_segments(
            "/mission/markers/probe_to_handle",
            points=np.zeros((1, 2, 3)),
            colors=(255, 170, 30),
            thickness=3.0,
            thickness_units="screen",
        )
        axis_points = np.stack(
            [np.zeros((3, 3)), 1.5 * np.eye(3)], axis=1
        )
        axis_colors = np.array(
            [
                [[255, 70, 70], [255, 70, 70]],
                [[70, 255, 70], [70, 255, 70]],
                [[70, 120, 255], [70, 120, 255]],
            ],
            dtype=np.uint8,
        )
        self.server.scene.add_line_segments(
            "/mission/lvlh_axes",
            points=axis_points,
            colors=axis_colors,
            thickness=3.0,
            thickness_units="screen",
        )

        self._setup_gui()
        self._render()

    def _setup_gui(self) -> None:
        self._sliders: list[Any] = []
        labels = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
        units = ("force", "force", "force", "torque", "torque", "torque")
        with self.server.gui.add_folder("Normalized action [-1, 1]"):
            for index, (label, unit) in enumerate(zip(labels, units)):
                slider = self.server.gui.add_slider(
                    label,
                    min=-1.0,
                    max=1.0,
                    step=0.05,
                    initial_value=0.0,
                    hint=f"Normalized {unit} command",
                )

                def update_action(event: Any, axis: int = index) -> None:
                    self.action[axis] = float(event.target.value)

                slider.on_update(update_action)
                self._sliders.append(slider)

            zero_button = self.server.gui.add_button("Zero force and torque")
            capture_button = self.server.gui.add_button("Request capture once")

        with self.server.gui.add_folder("Simulation"):
            run_checkbox = self.server.gui.add_checkbox(
                "Run continuously", initial_value=self._running
            )
            step_button = self.server.gui.add_button("Apply action for one step")
            reset_button = self.server.gui.add_button("Reset mission")
            speed_slider = self.server.gui.add_slider(
                "Playback speed",
                min=0.1,
                max=10.0,
                step=0.1,
                initial_value=1.0,
            )

        with self.server.gui.add_folder("Live sensor summary"):
            self._summary = self.server.gui.add_markdown("")
        with self.server.gui.add_folder("Complete sensor packet"):
            self._packet = self.server.gui.add_markdown("")

        @zero_button.on_click
        def _(_: Any) -> None:
            self.action[:6] = 0.0
            for slider in self._sliders:
                slider.value = 0.0

        @capture_button.on_click
        def _(_: Any) -> None:
            self._capture_requested = True

        @run_checkbox.on_update
        def _(_: Any) -> None:
            self._running = bool(run_checkbox.value)

        @step_button.on_click
        def _(_: Any) -> None:
            self._step_requested = True

        @reset_button.on_click
        def _(_: Any) -> None:
            self._reset_requested = True

        @speed_slider.on_update
        def _(_: Any) -> None:
            self._playback_speed = float(speed_slider.value)

    def run(self) -> None:
        """Serve the viewer until interrupted or the optional duration ends."""

        print(f"Manual rescue controls: http://localhost:{self.config.port}")
        print("Set an action in the browser, then Step or enable Run continuously.")
        next_step = time.perf_counter()
        try:
            while True:
                if self._reset_requested:
                    self.simulator.reset()
                    self._last_reward = None
                    self._last_info = {}
                    self._reset_requested = False
                    self._capture_requested = False

                now = time.perf_counter()
                continuous_step = self._running and now >= next_step
                if self._step_requested or continuous_step:
                    applied_action = self.action.copy()
                    applied_action[6] = 1.0 if self._capture_requested else -1.0
                    result = self.simulator.step(applied_action)
                    self._last_reward = result.reward
                    self._last_info = result.info
                    self._capture_requested = False
                    self._step_requested = False
                    next_step = now + (
                        self.simulator.config.control_dt_s / self._playback_speed
                    )
                    if result.terminated or result.truncated:
                        self._running = False

                self._render()
                if (
                    self.config.duration_s is not None
                    and self.simulator.time_s >= self.config.duration_s
                ):
                    break
                time.sleep(1.0 / 30.0)
        except KeyboardInterrupt:
            print("\nManual viewer stopped.")
        finally:
            self.close()

    def close(self) -> None:
        stop = getattr(self.server, "stop", None)
        if callable(stop):
            stop()

    def _render(self) -> None:
        _, data = self.simulator.visualization_handles
        rotation = np.asarray(data.frame.C_LI)
        self.scene.update(data, rotation=rotation)

        sensors = self.simulator.observe()
        probe = rotation @ np.asarray(sensors["capture_probe_position_world_m"])
        handle = rotation @ np.asarray(sensors["rescue_handle_position_world_m"])
        port = rotation @ np.asarray(sensors["spacecraft_port_position_world_m"])
        self._probe.position = tuple(probe)
        self._handle.position = tuple(handle)
        self._port.position = tuple(port)
        self._capture_line.points = np.asarray([[probe, handle]])

        dashboard_action = self.action.copy()
        dashboard_action[6] = 1.0 if self._capture_requested else -1.0
        self._summary.content = sensor_summary_markdown(
            sensors, dashboard_action, self._last_reward
        )
        serializable = sensor_packet_to_serializable(sensors)
        serializable["last_step_info"] = sensor_packet_to_serializable(self._last_info)
        self._packet.content = (
            "```json\n" + json.dumps(serializable, indent=2) + "\n```"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--auto-run", action="store_true")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--noisy-sensors", action="store_true")
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0.0:
        parser.error("--duration must be positive")

    simulator = RescueSimulator(
        RescueConfig(noisy_sensors=args.noisy_sensors)
    )
    app = ManualRescueApp(
        simulator,
        ManualViewerConfig(
            host=args.host,
            port=args.port,
            auto_run=args.auto_run,
            duration_s=args.duration,
        ),
    )
    app.run()


if __name__ == "__main__":
    main()
