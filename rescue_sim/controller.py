"""Simple controller used to demonstrate the simulator interface."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from rescue_sim.simulator import MissionPhase, RescueSimulator


class ProportionalDerivativeController:
    """Scripted rendezvous/capture/return controller.

    This is deliberately simple.  Its purpose is to demonstrate that an agent
    can consume the named sensor packet and produce actions through the same API
    that an MPC or reinforcement-learning policy would use.
    """

    def __init__(
        self,
        *,
        approach_position_gain_n_m: float = 0.20,
        approach_velocity_gain_n_s_m: float = 2.8,
        return_position_gain_n_m: float = 0.40,
        return_velocity_gain_n_s_m: float = 18.0,
        angular_rate_gain_nm_s_rad: float = 0.8,
    ) -> None:
        self.approach_position_gain = approach_position_gain_n_m
        self.approach_velocity_gain = approach_velocity_gain_n_s_m
        self.return_position_gain = return_position_gain_n_m
        self.return_velocity_gain = return_velocity_gain_n_s_m
        self.angular_rate_gain = angular_rate_gain_nm_s_rad

    def action(
        self,
        simulator: RescueSimulator,
        sensors: Mapping[str, Any],
    ) -> np.ndarray:
        config = simulator.config
        phase = MissionPhase(sensors["mission_phase"])

        if phase == MissionPhase.APPROACH:
            position_error = np.asarray(sensors["probe_to_handle_world_m"])
            velocity_error = np.asarray(sensors["relative_velocity_world_m_s"])
            force_n = (
                self.approach_position_gain * position_error
                + self.approach_velocity_gain * velocity_error
            )
            capture = (
                sensors["probe_handle_range_m"] < 0.8 * config.capture_range_m
                and np.linalg.norm(velocity_error) < 0.8 * config.capture_speed_m_s
            )
        elif phase == MissionPhase.CAPTURED:
            position_error = np.asarray(sensors["spacecraft_vector_world_m"])
            velocity_error = -np.asarray(sensors["rescue_velocity_world_m_s"])
            force_n = (
                self.return_position_gain * position_error
                + self.return_velocity_gain * velocity_error
            )
            capture = False
        else:
            force_n = np.zeros(3)
            capture = False

        omega_body = np.asarray(sensors["rescue_angular_velocity_body_rad_s"])
        torque_nm = -self.angular_rate_gain * omega_body

        action = np.zeros(simulator.ACTION_SIZE, dtype=np.float64)
        action[:3] = force_n / config.max_force_n
        action[3:6] = torque_nm / config.max_torque_nm
        action[6] = 1.0 if capture else -1.0
        return np.clip(action, -1.0, 1.0)
