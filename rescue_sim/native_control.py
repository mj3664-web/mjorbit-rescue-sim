"""Manual mission control in MuJoCo's native desktop viewer."""

from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np

from rescue_sim.simulator import (
    RescueConfig,
    RescueSimulator,
    sensor_packet_to_serializable,
)
from rescue_sim.viewer import RescueViewer, ViewerConfig


class NativeControlApp:
    """Map the native Control sliders to simulator actions and show telemetry."""

    def __init__(
        self,
        simulator: RescueSimulator,
        viewer_config: ViewerConfig,
        *,
        auto_run: bool = False,
        speed: float = 1.0,
        duration_s: float | None = None,
    ) -> None:
        self.simulator = simulator
        self.running = auto_run
        self.speed = speed
        self.duration_s = duration_s
        self._step_requested = False
        self._reset_requested = False
        self._print_requested = False
        self.last_reward: float | None = None
        self.viewer = RescueViewer(
            simulator,
            viewer_config,
            control_mode=True,
            key_callback=self._on_key,
        )

    def _on_key(self, keycode: int) -> None:
        key = chr(keycode).upper() if 0 <= keycode < 256 else ""
        if keycode == 32:
            self.running = not self.running
        elif key == "N":
            self._step_requested = True
        elif key == "R":
            self._reset_requested = True
        elif key == "P":
            self._print_requested = True

    def run(self) -> None:
        print("Native manual rescue control")
        print("  Expand the right-side Control panel for Fx/Fy/Fz/Tx/Ty/Tz/capture.")
        print("  Space: run/pause   N: one step   R: reset   P: print sensors")
        print("  Close the window or press Ctrl+C in this terminal to stop.")
        with self.viewer as viewer:
            next_step = time.perf_counter()
            try:
                while viewer.is_running:
                    if self._reset_requested:
                        self.simulator.reset()
                        self.last_reward = None
                        self._reset_requested = False

                    now = time.perf_counter()
                    if self._step_requested or (self.running and now >= next_step):
                        action = viewer.control_action()
                        result = self.simulator.step(action)
                        self.last_reward = result.reward
                        viewer.last_reward = result.reward
                        self._step_requested = False
                        next_step = now + self.simulator.config.control_dt_s / self.speed
                        if result.terminated or result.truncated:
                            self.running = False

                    sensors = self.simulator.observe()
                    if self._print_requested:
                        print(
                            json.dumps(
                                sensor_packet_to_serializable(sensors), indent=2
                            )
                        )
                        self._print_requested = False

                    action = viewer.control_action()
                    relative_speed = float(
                        np.linalg.norm(sensors["relative_velocity_world_m_s"])
                    )
                    reward_text = "--" if self.last_reward is None else f"{self.last_reward:.3f}"
                    viewer.set_overlay_lines(
                        (
                            f"{sensors['mission_phase'].upper()}  t={float(sensors['time_s']):.1f}s  "
                            f"reward={reward_text}",
                            f"probe={float(sensors['probe_handle_range_m']):.3f}m  "
                            f"rel_speed={relative_speed:.3f}m/s  "
                            f"port={float(sensors['distance_to_spacecraft_m']):.3f}m",
                            "action=" + np.array2string(
                                action, precision=2, suppress_small=True
                            ),
                        )
                    )
                    viewer.sync()
                    if (
                        self.duration_s is not None
                        and self.simulator.time_s >= self.duration_s
                    ):
                        break
                    time.sleep(1.0 / 60.0)
            except KeyboardInterrupt:
                print("\nNative control viewer stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto-run", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--noisy-sensors", action="store_true")
    parser.add_argument("--viewer-config", default=None)
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()
    if args.speed <= 0.0:
        parser.error("--speed must be positive")
    if args.duration is not None and args.duration <= 0.0:
        parser.error("--duration must be positive")

    simulator = RescueSimulator(
        RescueConfig(noisy_sensors=args.noisy_sensors)
    )
    app = NativeControlApp(
        simulator,
        ViewerConfig.from_toml(args.viewer_config),
        auto_run=args.auto_run,
        speed=args.speed,
        duration_s=args.duration,
    )
    try:
        app.run()
    finally:
        # Break the bound-method callback cycle before nanobind finalizes the
        # mjorbit model/data objects during interpreter shutdown.
        app.viewer.key_callback = None
        app.simulator.close()
        del app
        del simulator
        gc.collect()


if __name__ == "__main__":
    main()
