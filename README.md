# mjorbit astronaut-rescue simulator

This is a minimal end-to-end simulator for the mission:

1. a suited astronaut drifts away from a spacecraft in a 400 km circular orbit;
2. a free-flying rescue vehicle receives six-degree-of-freedom commands;
3. named simulated sensors report both vehicles' states;
4. the controller requests capture when the approach is safe;
5. a compliant connection engages; and
6. the combined vehicle returns to the spacecraft port.

The project uses mjorbit's coupled orbital and multibody dynamics. It deliberately
keeps capture geometry simple so that the action/sensor interface can be tested
before adding a robot arm, collision-aware approach, or reinforcement learning.

## Interface

Create and reset the environment:

```python
from rescue_sim import RescueSimulator

sim = RescueSimulator()
sensors = sim.reset()
```

Apply a normalized seven-element action:

```python
result = sim.step([fx, fy, fz, tx, ty, tz, capture])
sensors = result.sensors
```

The first three entries command force in the chief-centered inertial world frame.
The next three command body-frame torque. Values are normalized to `[-1, 1]` and
scaled by `max_force_n` and `max_torque_nm`. A final value greater than `0.5`
requests capture.

`step()` returns a `StepResult` with:

- `sensors`: named sensor values and derived navigation measurements;
- `reward`: a simple learning-compatible reward;
- `terminated`: mission success or escape;
- `truncated`: time limit reached; and
- `info`: applied physical commands and capture diagnostics.

Important sensor values include:

- `relative_position_world_m` and `relative_velocity_world_m_s`;
- `probe_to_handle_world_m` and `probe_handle_range_m`;
- both bodies' attitude and angular velocity;
- `spacecraft_vector_world_m` and `distance_to_spacecraft_m`; and
- `mission_phase`: `approach`, `captured`, or `complete`.

Set `RescueConfig(noisy_sensors=True)` to enable the noise declared in the MJCF
sensor definitions. Pass `False` for deterministic tests.

## Install

mjorbit currently documents Linux as its development platform and requires Python
3.11 or 3.12 plus a C++17 compiler. From this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

The dependency is pinned to the mjorbit revision used to build this prototype.

## Run the demonstration

```bash
rescue-demo --json-first-packet
```

or:

```bash
python -m rescue_sim.demo --json-first-packet
```

The included proportional-derivative controller reads only the returned sensor
packet and generates actions through the public interface. It is a demonstration
agent, not intended as a flight controller.

## Open the interactive viewer

Run the same mission in MuJoCo's interactive viewer:

```bash
rescue-view
```

or:

```bash
python -m rescue_sim.demo --viewer
```

The visualizer mirrors the mjorbit state into a display-only MuJoCo model; it
does not replace or modify the orbital simulation. The default playback rate is
4x. Use `--viewer-speed 1` for real time or another positive multiplier:

```bash
rescue-view --viewer-speed 2
```

On Windows, launch it from Ubuntu under WSLg. If GLFW is missing, install the
runtime package with `sudo apt install libglfw3`.

## Send actions manually and read sensors

Launch the browser-based manual control station:

```bash
rescue-control
```

Open `http://localhost:8080` if it does not open automatically. The control
station provides:

- normalized `Fx`, `Fy`, `Fz`, `Tx`, `Ty`, and `Tz` action sliders;
- a one-shot capture request button;
- single-step, continuous-run, speed, and reset controls;
- an interactive 3D mission view;
- a compact live navigation dashboard; and
- the complete sensor packet as JSON after every step.

The viewer starts paused. Set the six action values and click **Apply action for
one step** to send exactly one action and receive the resulting sensor values.
The capture command is momentary: click **Request capture once** before stepping.
Enable **Run continuously** to apply the current action at every control period.

Use a different port if 8080 is occupied:

```bash
rescue-control --port 8090
```

## Run tests

```bash
pytest
```

Tests verify that:

- actions change the simulated vehicle state;
- sensor packets are returned at every control step;
- malformed actions are rejected;
- unsafe capture commands are rejected; and
- the demonstration controller completes approach, capture, and return.

## Current fidelity and next steps

This first milestone models both the rescuer and astronaut as free rigid bodies.
Capture is a commanded compliant weld, engaged only below range and speed limits.
The spacecraft is the reference orbit with a port marker at the local origin.

Recommended next additions are:

1. astronaut tumble and capture-attitude constraints;
2. a physical gripper and backpack handle with contact forces;
3. keep-out zones around the helmet and spacecraft;
4. propellant bookkeeping and individual one-sided thrusters;
5. camera/range sensor occlusion and latency;
6. an MPPI baseline; and
7. a batched `mjorbit_warp` environment for PPO.
