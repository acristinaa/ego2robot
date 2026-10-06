import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CompositeObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import add_to_dict
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

BOWL_RADIUS = 0.06      #outer radius (m) - scaled-down version of the real 7 cm high bowl
BOWL_HEIGHT = 0.05
BOWL_WALL = 0.008
PLATE_RADIUS = 0.09
PLATE_HALF_THICK = 0.006


class BowlObject(CompositeObject):
    """Open bowl: a disc bottom plus a ring of thin wall segments (graspable rim)."""

    def __init__(self, name, n_walls=20, rgba=(0.15, 0.2, 0.35, 1.0)):
        self._name = name
        args = {"total_size": [BOWL_RADIUS, BOWL_RADIUS, BOWL_HEIGHT / 2], "name": name,
                "locations_relative_to_center": True, "obj_types": "all"}
        geoms = {}
        add_to_dict(geoms, geom_types="cylinder", geom_locations=(0, 0, -BOWL_HEIGHT / 2 + 0.004),
                    geom_quats=(1, 0, 0, 0), geom_sizes=(BOWL_RADIUS - BOWL_WALL / 2, 0.004),
                    geom_names="bottom", geom_rgbas=rgba, geom_frictions=(1.0, 0.005, 0.0001),
                    density=600)
        seg = 2 * np.pi * (BOWL_RADIUS - BOWL_WALL / 2) / n_walls / 2 * 1.08
        for i in range(n_walls):
            a = 2 * np.pi * i / n_walls
            r = BOWL_RADIUS - BOWL_WALL / 2
            add_to_dict(geoms, geom_types="box", geom_locations=(r * np.cos(a), r * np.sin(a), 0),
                        geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, a])), to="wxyz"),
                        geom_sizes=(BOWL_WALL / 2, seg, BOWL_HEIGHT / 2), geom_names=f"wall{i}",
                        geom_rgbas=rgba, geom_frictions=(1.5, 0.005, 0.0001), density=600)
        args.update(geoms)
        super().__init__(**args)


class BowlOnPlate(Lift):
    """Panda + table + bowl + plate. Success: bowl resting on the plate, gripper released."""

    def __init__(self, bowl_range=((-0.12, 0.06), (-0.22, -0.06)),
                 plate_range=((-0.08, 0.10), (0.06, 0.22)), **kwargs):
        self.bowl_range, self.plate_range = bowl_range, plate_range
        kwargs.setdefault("use_object_obs", False)
        super().__init__(**kwargs)

    def _load_model(self):
        super(Lift, self)._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction,
                           table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        self.bowl = BowlObject("bowl")
        self.plate = CylinderObject("plate", size=[PLATE_RADIUS, PLATE_HALF_THICK],
                                    rgba=[0.2, 0.3, 0.5, 1], density=3000, friction=(1.0, 0.005, 0.0001))
        self.placement_initializer = SequentialCompositeSampler(name="sampler")
        for obj, (xr, yr) in ((self.plate, self.plate_range), (self.bowl, self.bowl_range)):
            self.placement_initializer.append_sampler(UniformRandomSampler(
                name=f"{obj.name}_sampler", mujoco_objects=obj, x_range=xr, y_range=yr,
                rotation=None if obj is self.bowl else 0.0, ensure_object_boundary_in_range=False,
                ensure_valid_placement=True, reference_pos=self.table_offset, z_offset=0.002))
        self.model = ManipulationTask(mujoco_arena=arena,
                                      mujoco_robots=[r.robot_model for r in self.robots],
                                      mujoco_objects=[self.plate, self.bowl])

    def _setup_references(self):
        super(Lift, self)._setup_references()
        self.bowl_body_id = self.sim.model.body_name2id(self.bowl.root_body)
        self.plate_body_id = self.sim.model.body_name2id(self.plate.root_body)
        self.cube_body_id = self.bowl_body_id

    def _setup_observables(self):
        return super(Lift, self)._setup_observables()

    @property
    def bowl_pos(self):
        return np.array(self.sim.data.body_xpos[self.bowl_body_id])

    @property
    def plate_pos(self):
        return np.array(self.sim.data.body_xpos[self.plate_body_id])

    def _check_success(self):
        b, p = self.bowl_pos, self.plate_pos
        on_plate = (np.linalg.norm(b[:2] - p[:2]) < PLATE_RADIUS - 0.02 and
                    abs((b[2] - BOWL_HEIGHT / 2) - (p[2] + PLATE_HALF_THICK)) < 0.015)
        held = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.bowl)
        return bool(on_plate and not held)

    def reward(self, action=None):
        return float(self._check_success())


CAMERAS = ["agentview", "robot0_eye_in_hand", "birdview"]


def make_env(render=True, size=256, seed=None):
    import robosuite as suite
    from robosuite.controllers import load_controller_config
    if seed is not None:
        np.random.seed(seed)
    return BowlOnPlate(
        robots="Panda", controller_configs=load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False, has_offscreen_renderer=render, use_camera_obs=render,
        camera_names=CAMERAS, camera_heights=size, camera_widths=size,
        control_freq=20, horizon=600, ignore_done=True, reward_shaping=False)