"""Run a scripted action/sensor rescue demonstration."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from typing import Any

import numpy as np

from rescue_sim.controller import ProportionalDerivativeController
from rescue_sim.simulator import RescueConfig, RescueSimulator, sensor_packet_to_serializable


def main(*, viewer_default: bool = False) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--log-period", type=float, default=2.0)
    parser.add_argument("--noisy-sensors", action="store_true")
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=viewer_default,
        help="open an interactive MuJoCo window (default: off)",
    )
    parser.add_argument(
        "--viewer-speed",
        type=float,
        default=4.0,
        help="visualized simulation speed relative to wall time (default: 4)",
    )
    parser.add_argument(
        "--json-first-packet",
        action="store_true",
        help="print the complete initial sensor packet as JSON",
    )
    args = parser.parse_args()
    if args.viewer_speed <= 0.0:
        parser.error("--viewer-speed must be positive")

    simulator = RescueSimulator(
        RescueConfig(
            episode_duration_s=args.duration,
            noisy_sensors=args.noisy_sensors,
        )
    )
    controller = ProportionalDerivativeController()
    sensors = simulator.observe()
    if args.json_first_packet:
        print(json.dumps(sensor_packet_to_serializable(sensors), indent=2))

    print("mjorbit astronaut rescue: action -> physics -> sensors")
    print("action = [Fx, Fy, Fz, Tx, Ty, Tz, capture], normalized to [-1, 1]")
    print()
    viewer_context: Any = nullcontext(None)
    if args.viewer:
        # Imported lazily so headless users do not initialize GLFW.
        from rescue_sim.viewer import RescueViewer

        viewer_context = RescueViewer(simulator)
        print("Opening MuJoCo viewer. Close the window or press Ctrl+C to stop.")

    next_log = 0.0
    result = None
    with viewer_context as viewer:
        while True:
            wall_start = time.perf_counter()
            action = controller.action(simulator, sensors)
            result = simulator.step(action)
            sensors = result.sensors

            if sensors["time_s"] >= next_log or result.info["capture_event"]:
                print(
                    f"t={sensors['time_s']:6.1f}s  phase={sensors['mission_phase']:8s}  "
                    f"astronaut={sensors['probe_handle_range_m']:6.2f}m  "
                    f"spacecraft={sensors['distance_to_spacecraft_m']:6.2f}m  "
                    f"rel_speed={np.linalg.norm(sensors['relative_velocity_world_m_s']):5.3f}m/s  "
                    f"action={np.array2string(action, precision=2, suppress_small=True)}"
                )
                next_log += args.log_period

            if viewer is not None:
                if not viewer.sync():
                    print("Viewer closed; stopping the demonstration.")
                    break
                frame_period = simulator.config.control_dt_s / args.viewer_speed
                elapsed = time.perf_counter() - wall_start
                if elapsed < frame_period:
                    time.sleep(frame_period - elapsed)

            if result.terminated or result.truncated:
                break

    print()
    if result is not None and result.info["success"]:
        print("PASS: astronaut captured and returned to the spacecraft.")
    elif result is not None and result.info["escaped"]:
        print("FAIL: rescue vehicle exceeded the allowed range.")
    else:
        print("INCOMPLETE: episode time limit reached.")
    print(f"final phase: {sensors['mission_phase']}")
    print(f"capture attempts: {sensors['capture_attempts']}")
    print(f"last capture result: {sensors['last_capture_reason']}")


if __name__ == "__main__":
    main()
