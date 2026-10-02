"""Interactive MuJoCo visualization for the astronaut-rescue simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence
from xml.etree import ElementTree
import tomllib

import mujoco
import mujoco.viewer
import numpy as np

if TYPE_CHECKING:
    from rescue_sim.simulator import RescueSimulator


@dataclass(frozen=True)
class ViewerConfig:
    """Camera and UI settings for the interactive rescue viewer."""

    camera_lookat_m: tuple[float, float, float] = (5.5, 1.0, 0.5)
    camera_distance_m: float = 17.0
    camera_azimuth_deg: float = 135.0
    camera_elevation_deg: float = -22.0
    show_left_ui: bool = False
    show_right_ui: bool = True
    show_axes: bool = True
    show_markers: bool = True

    @classmethod
    def from_toml(cls, path: str | Path | None = None) -> ViewerConfig:
        """Load the shared native/manual viewer configuration."""

        config_path = (
            Path(path)
            if path is not None
            else Path(__file__).with_name("config") / "viewer.toml"
        )
        with config_path.open("rb") as file:
            raw = tomllib.load(file)
        camera = raw.get("camera", {})
        display = raw.get("display", {})
        lookat = tuple(float(value) for value in camera.get("lookat_m", (5.5, 1.0, 0.5)))
        if len(lookat) != 3:
            raise ValueError("camera.lookat_m must contain exactly three values")
        distance = float(camera.get("distance_m", 17.0))
        if distance <= 0.0:
            raise ValueError("camera.distance_m must be positive")
        return cls(
            camera_lookat_m=lookat,  # type: ignore[arg-type]
            camera_distance_m=distance,
            camera_azimuth_deg=float(camera.get("azimuth_deg", 135.0)),
            camera_elevation_deg=float(camera.get("elevation_deg", -22.0)),
            show_left_ui=bool(display.get("show_left_ui", False)),
            show_right_ui=bool(display.get("show_right_ui", True)),
            show_axes=bool(display.get("show_axes", True)),
            show_markers=bool(display.get("show_markers", True)),
        )

    def browser_camera_position_m(self) -> tuple[float, float, float]:
        """Convert the MuJoCo orbit-camera settings to a Viser eye position."""

        azimuth = np.deg2rad(self.camera_azimuth_deg)
        elevation = np.deg2rad(self.camera_elevation_deg)
        offset = self.camera_distance_m * np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                -np.sin(elevation),
            ]
        )
        return tuple(np.asarray(self.camera_lookat_m) + offset)


_MANUAL_SENSOR_NAMES = (
    "mission_phase_code",
    "probe_handle_range_m",
    "relative_speed_m_s",
    "distance_to_spacecraft_m",
    "last_reward",
    "capture_attempts",
)


def _viewer_model(*, control_mode: bool = False) -> mujoco.MjModel:
    """Compile a display-only copy of the rescue MJCF.

    Standard MuJoCo does not know mjorbit's custom top-level element, so it is
    removed from the visualization copy.  Physics continues to run exclusively
    in :class:`RescueSimulator`.
    """

    xml_path = Path(__file__).with_name("models") / "rescue_mission.xml"
    root = ElementTree.fromstring(xml_path.read_text(encoding="utf-8"))
    extension = root.find("mjorbit")
    if extension is not None:
        root.remove(extension)
    if control_mode:
        actuator = root.find("actuator")
        if actuator is None:
            raise RuntimeError("viewer model has no actuator section")
        for element in actuator:
            element.set("ctrlrange", "-1 1")
        ElementTree.SubElement(
            actuator,
            "motor",
            {
                "name": "capture_request",
                "joint": "rescuer_free",
                "gear": "0 0 0 0 0 0",
                "ctrlrange": "-1 1",
            },
        )
        sensor = root.find("sensor")
        if sensor is None:
            raise RuntimeError("viewer model has no sensor section")
        for name in _MANUAL_SENSOR_NAMES:
            ElementTree.SubElement(
                sensor,
                "user",
                {"name": name, "dim": "1", "needstage": "pos"},
            )
    xml = ElementTree.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml)


class RescueViewer:
    """Mirror a :class:`RescueSimulator` into MuJoCo's passive viewer."""

    def __init__(
        self,
        simulator: RescueSimulator,
        config: ViewerConfig | None = None,
        *,
        control_mode: bool = False,
        key_callback: Callable[[int], None] | None = None,
    ) -> None:
        self.simulator = simulator
        self.config = config or ViewerConfig.from_toml()
        self.control_mode = control_mode
        self.key_callback = key_callback
        self.model = _viewer_model(control_mode=control_mode)
        self.data = mujoco.MjData(self.model)
        self._handle: Any | None = None
        self._phase = ""

    def __enter__(self) -> RescueViewer:
        try:
            self._handle = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                key_callback=self.key_callback,
                show_left_ui=self.config.show_left_ui,
                show_right_ui=self.config.show_right_ui,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not open the MuJoCo viewer. Run inside WSLg/Ubuntu with "
                "a working DISPLAY and install libglfw3 if GLFW is unavailable."
            ) from exc

        self._handle.cam.lookat[:] = self.config.camera_lookat_m
        self._handle.cam.distance = self.config.camera_distance_m
        self._handle.cam.azimuth = self.config.camera_azimuth_deg
        self._handle.cam.elevation = self.config.camera_elevation_deg
        self.sync()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def is_running(self) -> bool:
        return bool(self._handle is not None and self._handle.is_running())

    def sync(self) -> bool:
        """Copy the current simulator pose into the window and redraw it."""

        if self._handle is None or not self._handle.is_running():
            return False

        state = self.simulator.render_state
        qpos = np.asarray(state["qpos"])
        qvel = np.asarray(state["qvel"])
        if qpos.shape != self.data.qpos.shape or qvel.shape != self.data.qvel.shape:
            raise RuntimeError("viewer and simulator state layouts do not match")

        with self._handle.lock():
            np.copyto(self.data.qpos, qpos)
            np.copyto(self.data.qvel, qvel)
            self.data.time = float(state["time_s"])
            mujoco.mj_forward(self.model, self.data)
            self._set_phase_appearance(str(state["mission_phase"]))
            if self.control_mode:
                self._write_manual_sensor_values()
        self._handle.sync()
        return True

    def control_action(self) -> np.ndarray:
        """Read the seven normalized sliders from a manual-control viewer."""

        if not self.control_mode or self.data.ctrl.shape != (7,):
            raise RuntimeError("viewer was not created in control mode")
        if self._handle is None:
            return np.asarray(self.data.ctrl).copy()
        with self._handle.lock():
            return np.asarray(self.data.ctrl).copy()

    def set_overlay_lines(self, lines: Sequence[str]) -> None:
        """Show short live telemetry labels inside the native 3D scene."""

        if self._handle is None:
            return
        positions = (
            np.array([4.0, 0.0, 3.0]),
            np.array([4.0, 0.0, 2.5]),
            np.array([4.0, 0.0, 2.0]),
        )
        colors = (
            np.array([0.25, 1.0, 0.45, 1.0], dtype=np.float32),
            np.array([0.3, 0.85, 1.0, 1.0], dtype=np.float32),
            np.array([1.0, 0.8, 0.25, 1.0], dtype=np.float32),
        )
        with self._handle.lock():
            scene = self._handle.user_scn
            count = min(len(lines), len(positions), scene.maxgeom)
            scene.ngeom = count
            for index in range(count):
                geom = scene.geoms[index]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LABEL,
                    np.array([0.18, 0.18, 0.18]),
                    positions[index],
                    np.eye(3).reshape(-1),
                    colors[index],
                )
                geom.label = str(lines[index])[:99]

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _set_phase_appearance(self, phase: str) -> None:
        if phase == self._phase:
            return
        probe_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "capture_probe"
        )
        if phase == "approach":
            self.model.site_rgba[probe_id] = (1.0, 0.8, 0.1, 1.0)
        elif phase == "captured":
            self.model.site_rgba[probe_id] = (0.2, 1.0, 0.3, 1.0)
        else:
            self.model.site_rgba[probe_id] = (0.2, 0.9, 1.0, 1.0)
        self._phase = phase

    def _write_manual_sensor_values(self) -> None:
        sensors = self.simulator.observe()
        reward = getattr(self, "last_reward", np.nan)
        values: Mapping[str, float] = {
            "mission_phase_code": float(
                {"approach": 0, "captured": 1, "complete": 2}[
                    str(sensors["mission_phase"])
                ]
            ),
            "probe_handle_range_m": float(sensors["probe_handle_range_m"]),
            "relative_speed_m_s": float(
                np.linalg.norm(sensors["relative_velocity_world_m_s"])
            ),
            "distance_to_spacecraft_m": float(
                sensors["distance_to_spacecraft_m"]
            ),
            "last_reward": float(reward),
            "capture_attempts": float(sensors["capture_attempts"]),
        }
        for name, value in values.items():
            sensor_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, name
            )
            address = int(self.model.sensor_adr[sensor_id])
            self.data.sensordata[address] = value
