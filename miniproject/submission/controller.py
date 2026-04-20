import numpy as np
from miniproject.simulation import MiniprojectSimulation


class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        # you may also implement your own turning controller
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)

        #johanne code
        self.odor_smooth = None
        self.alpha = 0.1  # smoothing factor for the low-pass filter
                
                
        self.vision_frames = []  # frames lisibles pour affichage




    def step(self, sim: MiniprojectSimulation):
        # implement your control algorithm here
        olfaction = sim.get_olfaction(sim.fly.name)

        #print(f"Olfaction: {olfaction}")  # for debugging purposes
        # get other observations as needed
        # drives = np.array([2.0, 2.0])  # replace with your control logic
        
        ################################### test joh ##################################


        ########################## odor tracking #########################
        if self.odor_smooth is None:
            self.odor_smooth = olfaction
        else:
            self.odor_smooth = (1 - self.alpha) * self.odor_smooth + self.alpha * olfaction


        drives = odor_to_drives(self.odor_smooth)
                
        ########################## vision obstacle avoidance #################
        ommatidia = sim.get_ommatidia_readouts(sim.fly.name)
        im = np.concatenate([
            sim.fly.retina.hex_pxls_to_human_readable(eye.max(-1), color_8bit=True)
            for eye in ommatidia
        ], axis=1)
        self.vision_frames.append(im)
        left_intensity = ommatidia[0].mean()
        right_intensity = ommatidia[1].mean()

        obstacle_left = 1 - left_intensity 
        obstacle_right = 1 - right_intensity

        obstacle_threshold = 0.5    # à tester
        avoidance_strength = 1.5    # à tester

        if obstacle_left > obstacle_threshold or obstacle_right > obstacle_threshold:
            if obstacle_left > obstacle_right:
                drives[1] += avoidance_strength
            else:
                drives[0] += avoidance_strength

        ###############################################################################

        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion

# Fonction odeur inspirée de la fontion du lab 4
def odor_to_drives(odor_intensities, attractive_gain=-500, aversive_gain=80): 
    n_sources = odor_intensities.shape[1]

    # Source attractive 
    attractive = np.average(
        odor_intensities[:, 0].reshape(2, 2), axis=0, weights=[9, 1]
    )
    attractive_bias = (
        attractive_gain * (attractive[0] - attractive[1]) / attractive.mean()
        if attractive.mean() != 0 else 0
    )

    # Source aversive (dimension 1 — optionnelle)
    aversive_bias = 0
    if n_sources >= 2:
        aversive = np.average(
            odor_intensities[:, 1].reshape(2, 2), axis=0, weights=[10, 0]
        )
        aversive_bias = (
            aversive_gain * (aversive[0] - aversive[1]) / aversive.mean()
            if aversive.mean() != 0 else 0
        )

    bias = attractive_bias + aversive_bias
    bias_norm = np.tanh(bias ** 2) * np.sign(bias)

    drives = np.ones(2)
    side = int(bias_norm > 0)
    drives[side] -= np.abs(bias_norm) * 0.8
    return drives