from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import mujoco

from handeye_mujoco import HandEyeSimulation


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "build/ur5e_ljv7080/model.xml"


def test_ur5e_tool0_sensor_contract():
    payload = json.loads((ROOT / "configs" / "ur5e_tool0_sensor.json").read_text())
    T_t_s = np.asarray(payload["T_tcp_sensor"], dtype=float)
    T_s_p = np.asarray(payload["T_sensor_physical"], dtype=float)

    np.testing.assert_allclose(T_t_s[:3, :3], np.diag([-1.0, 1.0, -1.0]))
    np.testing.assert_allclose(T_t_s[:3, 3], [0.0, 0.0, 151.0])
    np.testing.assert_allclose(T_s_p[:3, 3], [0.0, 0.0, 80.0])

    T_t_p = T_t_s.copy()
    T_t_p[:3, 3] = T_t_s[:3, :3] @ T_s_p[:3, 3] + T_t_s[:3, 3]
    np.testing.assert_allclose(T_t_p[:3, 3], [0.0, 0.0, 71.0])


def test_ur5e_model_keeps_visual_and_collision_meshes_separate():
    simulation = HandEyeSimulation(MODEL)
    names_by_group = {0: [], 1: []}
    for index in range(simulation.model.ngeom):
        group = int(simulation.model.geom_group[index])
        if group in names_by_group:
            names_by_group[group].append(
                mujoco.mj_id2name(
                    simulation.model, mujoco.mjtObj.mjOBJ_GEOM, index
                )
                or ""
            )
    assert sum(name.startswith("robot_visual_") for name in names_by_group[0]) == 7
    assert sum(name.startswith("robot_geom_") for name in names_by_group[1]) == 7
    visual_face_counts = []
    collision_face_counts = []
    for index in range(simulation.model.nmesh):
        name = (
            mujoco.mj_id2name(
                simulation.model, mujoco.mjtObj.mjOBJ_MESH, index
            )
            or ""
        )
        if name.startswith("robot_visual_mesh_"):
            visual_face_counts.append(int(simulation.model.mesh_facenum[index]))
        elif name in {
            "base",
            "shoulder",
            "upperarm",
            "forearm",
            "wrist1",
            "wrist2",
            "wrist3",
        }:
            collision_face_counts.append(int(simulation.model.mesh_facenum[index]))
    assert len(visual_face_counts) == len(collision_face_counts) == 7
    # Guards against accidentally loading only one material group from a DAE.
    assert sum(visual_face_counts) > 10 * sum(collision_face_counts)
    simulation.reset_home()
    assert simulation.collision_report().collision_free
