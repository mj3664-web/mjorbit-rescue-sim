"""Action/sensor environment for an orbital astronaut rescue mission.

The spacecraft is the mjorbit chief at the local-world origin.  A rescue vehicle
starts at the spacecraft and an astronaut drifts nearby.  The caller commands
the vehicle's force and torque and explicitly requests capture.  Once capture
conditions are safe, the simulator transfers the state into an otherwise
identical model with a compliant weld enabled, then the combined stack can be
flown back to the spacecraft.
"""

from __future__ import annotations

import enum
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mjorbit import MjoModel, OrbitInit, mjo_forward, mjo_step
from mjorbit.constants import GM_EARTH, R_EARTH
from mjorbit.rollout import mjo_get_state, mjo_set_state


class MissionPhase(str, enum.Enum):
    """Discrete mission state exposed to the controller."""

    APPROACH = "approach"
    CAPTURED = "captured"
    COMPLETE = "complete"


@dataclass(frozen=True)
class RescueConfig:
    """Static configuration for one rescue episode."""

    altitude_km: float = 400.0
    physics_dt_s: float = 0.01
    control_dt_s: float = 0.10
    episode_duration_s: float = 180.0
    max_force_n: float = 5.0
    max_torque_nm: float = 0.5
    capture_range_m: float = 0.20
    capture_speed_m_s: float = 0.15
    return_range_m: float = 0.60
    return_speed_m_s: float = 0.15
    escape_range_m: float = 100.0
    noisy_sensors: bool = True
    random_seed: int = 7

    def __post_init__(self) -> None:
        ratio = self.control_dt_s / self.physics_dt_s
        if self.physics_dt_s <= 0.0 or self.control_dt_s <= 0.0:
            raise ValueError("physics_dt_s and control_dt_s must be positive")
        if abs(ratio - round(ratio)) > 1.0e-9:
            raise ValueError("control_dt_s must be an integer multiple of physics_dt_s")
        if self.max_force_n <= 0.0 or self.max_torque_nm <= 0.0:
            raise ValueError("action limits must be positive")


@dataclass(frozen=True)
class StepResult:
    """Gym-like result returned by :meth:`RescueSimulator.step`."""

    sensors: Mapping[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]


class RescueSimulator:
    """Single-environment mjorbit astronaut-rescue simulator.

    Actions are seven normalized values in ``[-1, 1]``::

        [Fx, Fy, Fz, Tx, Ty, Tz, capture]

    Force axes are the chief-centered inertial world axes.  Torque axes are the
    rescue vehicle's body axes, following MuJoCo free-joint motor conventions.
    ``capture > 0.5`` requests capture; the request succeeds only when both the
    probe/handle range and relative speed are below their configured limits.
    """

    ACTION_SIZE = 7
    ACTION_NAMES = (
        "force_x",
        "force_y",
        "force_z",
        "torque_x",
        "torque_y",
        "torque_z",
        "capture",
    )

    def __init__(self, config: RescueConfig | None = None) -> None:
        self.config = config or RescueConfig()
        self._rng = np.random.default_rng(self.config.random_seed)
        self._free_model, self._captured_model = self._compile_models()
        self._model = self._free_model
        self._orbit = self._make_orbit()
        self._data = self._model.make_data(
            orbit=self._orbit,
            rng_seed=self.config.random_seed,
        )
        self._decimation = int(round(self.config.control_dt_s / self.config.physics_dt_s))
        self._phase = MissionPhase.APPROACH
        self._capture_attempts = 0
        self._last_capture_reason = "not_requested"
        self._last_action = np.zeros(self.ACTION_SIZE, dtype=np.float64)
        self._actuator_ids = self._find_actuators(self._model)
        self.reset()

    @property
    def phase(self) -> MissionPhase:
        return self._phase

    @property
    def time_s(self) -> float:
        return float(self._data.time)

    @property
    def action_spec(self) -> Mapping[str, Any]:
        """Machine-readable description of the action interface."""

        return {
            "shape": (self.ACTION_SIZE,),
            "range": (-1.0, 1.0),
            "names": self.ACTION_NAMES,
            "force_scale_n": self.config.max_force_n,
            "torque_scale_nm": self.config.max_torque_nm,
            "capture_threshold": 0.5,
        }

    @property
    def render_state(self) -> Mapping[str, Any]:
        """Return a copy of the MuJoCo state needed by visualization clients.

        The simulator deliberately keeps the native mjorbit model and data
        handles private.  A viewer can mirror these arrays into a standard
        MuJoCo model without being able to mutate the mission dynamics.
        """

        return {
            "time_s": self.time_s,
            "qpos": np.asarray(self._data.qpos).copy(),
            "qvel": np.asarray(self._data.qvel).copy(),
            "mission_phase": self._phase.value,
        }

    @property
    def visualization_handles(self) -> tuple[MjoModel, Any]:
        """Return live model/data handles for read-only visualization.

        These handles change when capture switches to the welded model.  A
        visualization client should request them again on every refresh and
        must not mutate either object.
        """

        return self._model, self._data

    def reset(
        self,
        *,
        astronaut_position_lvlh_m: np.ndarray | None = None,
        astronaut_velocity_lvlh_m_s: np.ndarray | None = None,
    ) -> Mapping[str, Any]:
        """Reset the episode and return the initial sensor packet."""

        self._model = self._free_model
        self._data = self._model.make_data(
            orbit=self._orbit,
            rng_seed=self.config.random_seed,
        )
        self._actuator_ids = self._find_actuators(self._model)
        self._phase = MissionPhase.APPROACH
        self._capture_attempts = 0
        self._last_capture_reason = "not_requested"
        self._last_action.fill(0.0)

        astronaut_position_lvlh_m = np.asarray(
            [12.0, 2.0, 1.0]
            if astronaut_position_lvlh_m is None
            else astronaut_position_lvlh_m,
            dtype=np.float64,
        )
        astronaut_velocity_lvlh_m_s = np.asarray(
            [0.0, 0.015, -0.005]
            if astronaut_velocity_lvlh_m_s is None
            else astronaut_velocity_lvlh_m_s,
            dtype=np.float64,
        )
        if astronaut_position_lvlh_m.shape != (3,):
            raise ValueError("astronaut_position_lvlh_m must have shape (3,)")
        if astronaut_velocity_lvlh_m_s.shape != (3,):
            raise ValueError("astronaut_velocity_lvlh_m_s must have shape (3,)")

        # qpos layout: rescuer free joint [0:7], astronaut free joint [7:14].
        self._data.qpos[0:3] = self._data.world_position_from_lvlh([0.0, 0.0, 0.0])
        self._data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self._data.qpos[7:10] = self._data.world_position_from_lvlh(
            astronaut_position_lvlh_m
        )
        self._data.qpos[10:14] = [1.0, 0.0, 0.0, 0.0]

        self._data.qvel[0:3] = self._data.world_velocity_from_lvlh(
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        )
        self._data.qvel[3:6] = [0.0, 0.0, 0.0]
        self._data.qvel[6:9] = self._data.world_velocity_from_lvlh(
            astronaut_position_lvlh_m,
            astronaut_velocity_lvlh_m_s,
        )
        self._data.qvel[9:12] = [0.0, 0.0, 0.0]
        self._data.ctrl[:] = 0.0
        mjo_forward(self._model, self._data)
        return self.observe()

    def observe(self) -> Mapping[str, Any]:
        """Return a named sensor packet without advancing physics."""

        raw = self._data.sensors.measure_all(
            noisy=self.config.noisy_sensors,
            rng=self._rng,
        )
        rescue_pos = raw["rescue_position_world_m"].copy()
        astronaut_pos = raw["astronaut_position_world_m"].copy()
        rescue_vel = raw["rescue_velocity_world_m_s"].copy()
        astronaut_vel = raw["astronaut_velocity_world_m_s"].copy()
        relative_position = astronaut_pos - rescue_pos
        relative_velocity = astronaut_vel - rescue_vel
        relative_range = float(np.linalg.norm(relative_position))
        if relative_range > 1.0e-12:
            closing_speed = -float(relative_position @ relative_velocity) / relative_range
        else:
            closing_speed = 0.0

        probe = raw["capture_probe_position_world_m"].copy()
        handle = raw["rescue_handle_position_world_m"].copy()
        probe_to_handle = handle - probe
        port = raw["spacecraft_port_position_world_m"].copy()

        return {
            "time_s": self.time_s,
            "mission_phase": self._phase.value,
            "rescue_position_world_m": rescue_pos,
            "rescue_velocity_world_m_s": rescue_vel,
            "rescue_attitude_wxyz": raw["rescue_attitude_wxyz"].copy(),
            "rescue_angular_velocity_body_rad_s": raw[
                "rescue_angular_velocity_body_rad_s"
            ].copy(),
            "astronaut_position_world_m": astronaut_pos,
            "astronaut_velocity_world_m_s": astronaut_vel,
            "astronaut_attitude_wxyz": raw["astronaut_attitude_wxyz"].copy(),
            "astronaut_angular_velocity_body_rad_s": raw[
                "astronaut_angular_velocity_body_rad_s"
            ].copy(),
            "relative_position_world_m": relative_position,
            "relative_velocity_world_m_s": relative_velocity,
            "range_m": relative_range,
            "closing_speed_m_s": closing_speed,
            "capture_probe_position_world_m": probe,
            "rescue_handle_position_world_m": handle,
            "probe_to_handle_world_m": probe_to_handle,
            "probe_handle_range_m": float(np.linalg.norm(probe_to_handle)),
            "spacecraft_port_position_world_m": port,
            "spacecraft_vector_world_m": port - rescue_pos,
            "distance_to_spacecraft_m": float(np.linalg.norm(port - rescue_pos)),
            "capture_attempts": self._capture_attempts,
            "last_capture_reason": self._last_capture_reason,
        }

    def step(self, action: np.ndarray | list[float]) -> StepResult:
        """Apply one action for one control interval and return sensor values."""

        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (self.ACTION_SIZE,):
            raise ValueError(f"action must have shape ({self.ACTION_SIZE},)")
        if not np.all(np.isfinite(action_array)):
            raise ValueError("action must contain only finite values")
        action_array = np.clip(action_array, -1.0, 1.0)
        self._last_action = action_array.copy()

        self._apply_action(action_array)
        for _ in range(self._decimation):
            mjo_step(self._model, self._data)

        sensors = self.observe()
        capture_event = False
        if self._phase == MissionPhase.APPROACH and action_array[6] > 0.5:
            self._capture_attempts += 1
            capture_event = self._try_capture(sensors)
            sensors = self.observe()

        success = False
        if self._phase == MissionPhase.CAPTURED:
            speed = float(np.linalg.norm(sensors["rescue_velocity_world_m_s"]))
            if (
                sensors["distance_to_spacecraft_m"] <= self.config.return_range_m
                and speed <= self.config.return_speed_m_s
            ):
                self._phase = MissionPhase.COMPLETE
                success = True
                sensors = self.observe()

        escaped = sensors["distance_to_spacecraft_m"] > self.config.escape_range_m
        terminated = bool(success or escaped)
        truncated = bool(self.time_s >= self.config.episode_duration_s and not terminated)
        reward = self._reward(sensors, capture_event=capture_event, success=success, escaped=escaped)
        info = {
            "phase": self._phase.value,
            "capture_event": capture_event,
            "capture_reason": self._last_capture_reason,
            "success": success,
            "escaped": bool(escaped),
            "applied_force_world_n": action_array[:3] * self.config.max_force_n,
            "applied_torque_body_nm": action_array[3:6] * self.config.max_torque_nm,
        }
        return StepResult(sensors, reward, terminated, truncated, info)

    def close(self) -> None:
        """Release references held by the simulator."""

        self._data = None  # type: ignore[assignment]

    def _make_orbit(self) -> OrbitInit:
        radius_km = R_EARTH + self.config.altitude_km
        return OrbitInit(
            R_eci=[radius_km, 0.0, 0.0],
            V_eci=[0.0, np.sqrt(GM_EARTH / radius_km), 0.0],
        )

    def _compile_models(self) -> tuple[MjoModel, MjoModel]:
        xml_path = Path(__file__).with_name("models") / "rescue_mission.xml"
        xml = xml_path.read_text(encoding="utf-8")
        models: list[MjoModel] = []
        for active in ("false", "true"):
            variant = xml.replace('active="false"', f'active="{active}"')
            with tempfile.NamedTemporaryFile(
                suffix=".xml", mode="w", encoding="utf-8", delete=False
            ) as file:
                file.write(variant)
                temporary_path = Path(file.name)
            try:
                models.append(
                    MjoModel.from_xml_path(
                        str(temporary_path),
                        mj_timestep=self.config.physics_dt_s,
                    )
                )
            finally:
                temporary_path.unlink(missing_ok=True)
        return models[0], models[1]

    @staticmethod
    def _find_actuators(model: MjoModel) -> np.ndarray:
        # The XML fixes the six control channels in force-then-torque order.
        if int(model.nu) != 6:
            raise RuntimeError(f"expected six vehicle actuators, found {model.nu}")
        return np.arange(6, dtype=int)

    def _apply_action(self, action: np.ndarray) -> None:
        self._data.ctrl[self._actuator_ids[:3]] = action[:3] * self.config.max_force_n
        self._data.ctrl[self._actuator_ids[3:]] = action[3:6] * self.config.max_torque_nm

    def _try_capture(self, sensors: Mapping[str, Any]) -> bool:
        probe_range = float(sensors["probe_handle_range_m"])
        relative_speed = float(np.linalg.norm(sensors["relative_velocity_world_m_s"]))
        if probe_range > self.config.capture_range_m:
            self._last_capture_reason = "probe_out_of_range"
            return False
        if relative_speed > self.config.capture_speed_m_s:
            self._last_capture_reason = "relative_speed_too_high"
            return False

        old_ctrl = np.asarray(self._data.ctrl).copy()
        state = mjo_get_state(self._model, self._data)
        captured_data = self._captured_model.make_data(
            orbit=self._orbit,
            rng_seed=self.config.random_seed,
        )
        mjo_set_state(self._captured_model, captured_data, state)
        np.copyto(captured_data.ctrl, old_ctrl)
        mjo_forward(self._captured_model, captured_data)

        self._model = self._captured_model
        self._data = captured_data
        self._actuator_ids = self._find_actuators(self._model)
        self._phase = MissionPhase.CAPTURED
        self._last_capture_reason = "capture_succeeded"
        return True

    def _reward(
        self,
        sensors: Mapping[str, Any],
        *,
        capture_event: bool,
        success: bool,
        escaped: bool,
    ) -> float:
        if self._phase == MissionPhase.APPROACH:
            distance_cost = float(sensors["probe_handle_range_m"])
        else:
            distance_cost = float(sensors["distance_to_spacecraft_m"])
        effort = float(np.dot(self._last_action[:6], self._last_action[:6]))
        reward = -0.02 * distance_cost - 0.001 * effort
        if capture_event:
            reward += 25.0
        if success:
            reward += 100.0
        if escaped:
            reward -= 100.0
        return reward


def sensor_packet_to_serializable(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Convert NumPy sensor values to JSON-friendly Python values."""

    serializable: dict[str, Any] = {}
    for key, value in packet.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, np.generic):
            serializable[key] = value.item()
        else:
            serializable[key] = value
    return serializable
