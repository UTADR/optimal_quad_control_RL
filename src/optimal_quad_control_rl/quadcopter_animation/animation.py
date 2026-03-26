import time
from collections import deque

import numpy as np
import rerun as rr
import scipy.spatial.transform as spt

_TRAIL_MAX_LEN = 2000  # positions kept per drone

_GATE_SIZE = 1.5  # gate side length in metres
_DRONE_ARM_RADIUS = 0.08  # arm half-span (m)


def _gate_corners(pos: np.ndarray, yaw: float) -> np.ndarray:
    """Closed square gate corners in world frame (5 pts, last == first)."""
    h = _GATE_SIZE / 2
    # Gate normal is along gate-frame X; gate lies in gate-frame YZ plane.
    local = np.array([[0, h, h], [0, -h, h], [0, -h, -h], [0, h, -h], [0, h, h]])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (R @ local.T).T + pos


def _drone_body_strips(r: float = _DRONE_ARM_RADIUS):
    """LineStrips in drone body frame: 4 arms + 4 propeller circles."""
    motor_pos = np.array([[r, r, 0.0], [-r, r, 0.0], [-r, -r, 0.0], [r, -r, 0.0]])
    strips = []
    # Arms from body centre to each motor
    for mp in motor_pos:
        strips.append([[0.0, 0.0, 0.0], mp.tolist()])
    # Propeller circles in body XY plane
    prop_r = 2.0 * r / 3.0
    angles = np.linspace(0.0, 2.0 * np.pi, 33)
    for mp in motor_pos:
        circle = np.column_stack(
            [
                mp[0] + prop_r * np.cos(angles),
                mp[1] + prop_r * np.sin(angles),
                np.zeros(33),
            ]
        )
        strips.append(circle.tolist())
    return strips


_DRONE_STRIPS = _drone_body_strips()


def _zero_state():
    return {"x": 0.0, "y": 0.0, "z": 0.0, "phi": 0.0, "theta": 0.0, "psi": 0.0}


def view(
    get_drone_state=_zero_state,
    fps: float = 100,
    gate_pos=(),
    gate_yaw=(),
    waypoints=(),
):
    """Real-time quadrotor visualisation via Rerun.

    Parameters
    ----------
    get_drone_state:
        Callable returning a dict with keys ``x, y, z, phi, theta, psi``
        (and optionally ``u1–u4``).  Values may be scalars (single drone)
        or equal-length arrays (multiple drones).
    fps:
        Simulation rate; controls the sleep between frames.
    gate_pos:
        Sequence of (x, y, z) gate centre positions.
    gate_yaw:
        Sequence of gate yaw angles (radians), one per gate.
    waypoints:
        Sequence of (x, y, z) waypoint positions.  Rendered as static
        point markers.
    """
    rr.init("quadrotor_sim", spawn=True)

    # Gates are static — logged once
    for i, (gpos, gyaw) in enumerate(zip(gate_pos, gate_yaw)):
        corners = _gate_corners(np.asarray(gpos, dtype=float), float(gyaw))
        rr.log(
            f"world/gate/{i}",
            rr.LineStrips3D([corners], colors=[[0, 140, 255]]),
            static=True,
        )

    # Waypoints are static — logged once as point markers
    if len(waypoints):
        wp_arr = np.asarray(waypoints, dtype=float)
        rr.log(
            "world/waypoints",
            rr.Points3D(wp_arr, radii=0.08, colors=[[0, 220, 100]]),
            static=True,
        )

    logged_drone_indices: set[int] = set()
    trails: dict[int, deque] = {}
    sim_time = 0.0
    dt = float(1.0 / fps)

    try:
        while True:
            t_loop_start = time.monotonic()
            state = get_drone_state()

            positions = np.column_stack(
                [np.asarray(state[k], dtype=float) for k in ("x", "y", "z")]
            )

            eulers = np.column_stack(
                [np.asarray(state[k], dtype=float) for k in ("phi", "theta", "psi")]
            )
            orientations = spt.Rotation.from_euler("xyz", eulers).as_quat(
                canonical=True
            )

            rr.set_time("sim_time", timestamp=sim_time)

            for i, (ps, qs) in enumerate(zip(positions, orientations)):
                # Log body geometry once as static under each drone entity
                if i not in logged_drone_indices:
                    rr.log(
                        f"world/drone/{i}/pose/body",
                        rr.LineStrips3D(_DRONE_STRIPS, colors=[[255, 0, 0]]),
                        static=True,
                    )
                    logged_drone_indices.add(i)
                    trails[i] = deque(maxlen=_TRAIL_MAX_LEN)

                rr.log(
                    f"world/drone/{i}/pose",
                    rr.Transform3D(translation=ps, rotation=rr.Quaternion(xyzw=qs)),
                )

                trails[i].append(ps.tolist())
                if len(trails[i]) >= 2:
                    rr.log(
                        f"world/drone/{i}/trail",
                        rr.LineStrips3D([list(trails[i])], colors=[[255, 180, 0]]),
                    )

            sim_time += dt
            elapsed = time.monotonic() - t_loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        pass
