import numpy as np
import mujoco
from dm_control.utils import transformations
from scipy.special import k0e
from functools import cached_property, cache


def compute_log_concentration(
    xy,
    source_xy,
    flow_velocity,
    diffusivity,
    decay,
    emission_rate,
):
    r = xy - source_xy
    kappa_r = np.sqrt(
        flow_velocity @ flow_velocity / (4 * diffusivity * diffusivity)
        + decay / diffusivity
    ) * np.linalg.norm(r, axis=-1)
    log_c = (
        np.log(emission_rate / (2 * np.pi * diffusivity))
        + (r @ flow_velocity) / (2 * diffusivity)
        + np.log(k0e(kappa_r))
        - kappa_r
    )
    return log_c


class BananaSliceMixin:
    banana_xy: tuple[float, float]

    @cached_property
    def _bananas(self):
        return []

    def add_banana_slice(
        self,
        pos,
        radius=1,
        height=0.5,
        decay=500.0,
        diffusivity=20000.0,
        emission_rate=1.0,
        flow_velocity=(0.0, 0.0),
    ):
        normal = (0, 0, 1)
        try:
            x, y = pos
            if hasattr(self, "get_height"):
                z = self.get_height(x, y)
            else:
                z = 0
            if hasattr(self, "get_normal"):
                normal = tuple(self.get_normal(x, y))
        except ValueError:
            x, y, z = pos

        body = self.mjcf_root.worldbody.add(
            "body",
            name="banana_slice",
            pos=(x, y, z),
            zaxis=normal,
            mocap=True,
        )
        body.add(
            "geom",
            name="peel",
            type="cylinder",
            size=(radius, height / 2),
            rgba=(1, 1, 0, 1),
        )
        body.add(
            "geom",
            name="flesh",
            type="cylinder",
            size=(radius * 0.8, height / 2 * 1.05),
            rgba=(0.9, 0.84, 0.68, 1),
        )
        self._bananas.append(body)
        self.banana_xy = (x, y)
        self.flow_velocity = np.array(flow_velocity)
        self.decay = decay
        self.diffusivity = diffusivity
        self.emission_rate = emission_rate

    def get_olfaction(self, sensor_positions: np.ndarray, log=False):
        """Returns olfaction signals based on distance to the banana slice."""
        if not hasattr(self, "banana_xy"):
            raise ValueError("Banana slice not added to the world.")

        log_concentration = compute_log_concentration(
            sensor_positions[..., :2],
            self.banana_xy,
            self.flow_velocity,
            self.diffusivity,
            self.decay,
            self.emission_rate,
        )[..., None]

        if log:
            return log_concentration

        return np.exp(log_concentration)
    
    @cache
    def _get_banana_mocap_id(self, sim, banana_idx=0):
        banana = self._bananas[banana_idx]
        return sim.mj_model.body_mocapid[
            mujoco.mj_name2id(
                sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, banana.full_identifier
            )
        ]

    def set_banana_pose(self, sim, pos, euler_ZYX=None, banana_idx=0):
        mocap_id = self._get_banana_mocap_id(sim, banana_idx)
        sim.mj_data.mocap_pos[mocap_id] = pos
        self.banana_xy = (pos[0], pos[1])
        if euler_ZYX is not None:
            quat = transformations.euler_to_quat(euler_ZYX, "ZYX")
            sim.mj_data.mocap_quat[mocap_id] = quat
