"""Run a scripted action/sensor rescue demonstration."""

from __future__ import annotations

import argparse
import json

import numpy as np

from rescue_sim.controller import ProportionalDerivativeController
from rescue_sim.simulator import RescueConfig, RescueSimulator, sensor_packet_to_serializable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--log-period", type=float, default=2.0)
    parser.add_argument("--noisy-sensors", action="store_true")
    parser.add_argument(
        "--json-first-packet",
        action="store_true",
        help="print the complete initial sensor packet as JSON",
    )
    args = parser.parse_args()

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
    next_log = 0.0
    result = None
    while True:
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

