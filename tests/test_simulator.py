from __future__ import annotations

import numpy as np
import pytest

from rescue_sim import ProportionalDerivativeController, RescueConfig, RescueSimulator


@pytest.fixture
def simulator() -> RescueSimulator:
    return RescueSimulator(
        RescueConfig(
            noisy_sensors=False,
            episode_duration_s=180.0,
        )
    )


def test_action_changes_velocity_and_returns_sensor_packet(simulator: RescueSimulator) -> None:
    initial = simulator.observe()
    action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    result = simulator.step(action)

    assert result.sensors["time_s"] == pytest.approx(0.1)
    assert result.sensors["rescue_velocity_world_m_s"][0] > initial[
        "rescue_velocity_world_m_s"
    ][0]
    assert result.info["applied_force_world_n"][0] == pytest.approx(5.0)
    assert result.sensors["mission_phase"] == "approach"


def test_invalid_action_is_rejected(simulator: RescueSimulator) -> None:
    with pytest.raises(ValueError, match="shape"):
        simulator.step(np.zeros(6))
    with pytest.raises(ValueError, match="finite"):
        simulator.step(np.array([0, 0, 0, 0, 0, 0, np.nan]))


def test_capture_rejected_when_out_of_range(simulator: RescueSimulator) -> None:
    result = simulator.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
    assert not result.info["capture_event"]
    assert result.info["capture_reason"] == "probe_out_of_range"
    assert result.sensors["capture_attempts"] == 1


def test_scripted_controller_completes_rescue(simulator: RescueSimulator) -> None:
    controller = ProportionalDerivativeController()
    sensors = simulator.observe()
    result = None
    while simulator.time_s < simulator.config.episode_duration_s:
        result = simulator.step(controller.action(simulator, sensors))
        sensors = result.sensors
        if result.terminated or result.truncated:
            break

    assert result is not None
    assert result.info["success"]
    assert sensors["mission_phase"] == "complete"
    assert sensors["last_capture_reason"] == "capture_succeeded"

