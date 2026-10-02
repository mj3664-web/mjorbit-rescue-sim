from __future__ import annotations

import numpy as np

from rescue_sim import RescueConfig, RescueSimulator
from rescue_sim.viewer import ViewerConfig, _viewer_model


def test_viewer_model_matches_simulator_state_layout() -> None:
    simulator = RescueSimulator(RescueConfig(noisy_sensors=False))
    model = _viewer_model()
    state = simulator.render_state

    assert model.nq == np.asarray(state["qpos"]).size
    assert model.nv == np.asarray(state["qvel"]).size
    assert model.ngeom >= 9


def test_render_state_is_a_non_mutating_copy() -> None:
    simulator = RescueSimulator(RescueConfig(noisy_sensors=False))
    state = simulator.render_state
    original_position = simulator.observe()["rescue_position_world_m"].copy()

    state["qpos"][0] += 1000.0

    np.testing.assert_allclose(
        simulator.observe()["rescue_position_world_m"], original_position
    )


def test_shared_viewer_configuration_loads_and_maps_browser_camera() -> None:
    config = ViewerConfig.from_toml()
    position = np.asarray(config.browser_camera_position_m())

    assert position.shape == (3,)
    np.testing.assert_allclose(
        np.linalg.norm(position - np.asarray(config.camera_lookat_m)),
        config.camera_distance_m,
    )


def test_native_control_model_exposes_seven_actions_and_derived_sensors() -> None:
    model = _viewer_model(control_mode=True)

    assert model.nu == 7
    assert model.actuator("capture_request").id == 6
    for name in (
        "mission_phase_code",
        "probe_handle_range_m",
        "relative_speed_m_s",
        "distance_to_spacecraft_m",
        "last_reward",
        "capture_attempts",
    ):
        assert model.sensor(name).id >= 0
