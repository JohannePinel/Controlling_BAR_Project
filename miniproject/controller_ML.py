import numpy as np
import torch
from pathlib import Path
from collections import deque
from miniproject.simulation import MiniprojectSimulation
from model import ObstacleNet 

# ========= Odor =========
OLFACTION_RATE = 10
DISCARD_VALUE = 123456789
ALPHA = 0.1   
ODOR_DETECTION_THRESHOLD = 1e-8

# ========= Vision =========
VISION_RATE = 100

# ========= Model =========
WINDOW = 5
DANGER_THRESHOLD = 0.5


def render_ommatidia(sim):
    """Convert ommatidia readouts to a displayable RGB image."""
    ommatidia_brut = sim.get_ommatidia_readouts(sim.fly.name)

    # Convert the two eyes to human-readable images
    left_eye = sim.fly.retina.hex_pxls_to_human_readable(
        ommatidia_brut[0].max(-1), color_8bit=True # Corrected: ommatidia[0] for left eye
    )
    right_eye = sim.fly.retina.hex_pxls_to_human_readable(
        ommatidia_brut[1].max(-1), color_8bit=True # Corrected: ommatidia[1] for right eye
    )
    vision_img = np.concatenate([left_eye, right_eye], axis=1)
    vision_img = np.stack([vision_img] * 3, axis=-1) # No difference but adds a 3rd argument (so the np.pad doesn't crash)

    return vision_img, ommatidia_brut

def odor_to_drives(odor_intensities, attractive_gain=-500, aversive_gain=80): 
    n_sources = odor_intensities.shape[1]
    attractive = np.average(
        odor_intensities[:, 0].reshape(2, 2), axis=0, weights=[9, 1])
    attractive_bias = (
        attractive_gain * (attractive[0] - attractive[1]) / attractive.mean()
        if attractive.mean() != 0 else 0)
    aversive_bias = 0
    if n_sources >= 2:
        aversive = np.average(
            odor_intensities[:, 1].reshape(2, 2), axis=0, weights=[10, 0])
        aversive_bias = (
            aversive_gain * (aversive[0] - aversive[1]) / aversive.mean()
            if aversive.mean() != 0 else 0)

    bias = attractive_bias + aversive_bias
    bias_norm = np.tanh(bias ** 2) * np.sign(bias)

    drives = np.ones(2)
    side = int(bias_norm > 0)
    drives[side] -= np.abs(bias_norm) * 0.8
    return drives


class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)

        """ ODOR """
        self.odor_smooth = None
        self.odor_mode_enabled = True  # odor_mode activé par défaut

        """ WALKING """
        self.speed = 1.0

        """ MODELE """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ObstacleNet().to(self.device)
        model_path = Path(__file__).parent / "obstacle_net.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.ommatidia_buffer = deque(maxlen=WINDOW)


    def step(self, sim: MiniprojectSimulation):

        """ ODOR """
        olfaction = sim.get_olfaction(sim.fly.name)
        if self.odor_smooth is None:
            self.odor_smooth = olfaction
        else:
            self.odor_smooth = (1 - ALPHA) * self.odor_smooth + ALPHA * olfaction
        odor_drives = odor_to_drives(self.odor_smooth) * 3 if self.odor_mode_enabled else np.ones(2)


        """ VISION + MODEL """
        _, ommatidia = render_ommatidia(sim)
        self.ommatidia_buffer.append(ommatidia)

        gain_left, gain_right = 1.0, 1.0
        if len(self.ommatidia_buffer) == WINDOW:
            window = np.stack(list(self.ommatidia_buffer))  # (5, 2, 721, 2)
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 5, 2, 721, 2)

            with torch.no_grad():
                danger, pred_gains = self.model(x)

            if danger.item() > DANGER_THRESHOLD:
                gain_left, gain_right = pred_gains[0].cpu().numpy()
                odor_drives = np.ones(2)


        """ INSTRUCTIONS TO BODY """
        drives = self.speed * odor_drives * np.array([gain_left, gain_right])
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion


