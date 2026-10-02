from __future__ import annotations

import numpy as np

from rescue_sim import RescueConfig, RescueSimulator
from rescue_sim.manual import sensor_summary_markdown


def test_sensor_summary_contains_action_and_navigation_values() -> None:
    simulator = RescueSimulator(RescueConfig(noisy_sensors=False))
    sensors = simulator.observe()
    action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

    summary = sensor_summary_markdown(sensors, action, reward=None)

    assert "Mission phase" in summary
    assert "Probe → handle" in summary
    assert "Relative speed" in summary
    assert "Applied action" in summary
    assert sensors["mission_phase"] in summary
