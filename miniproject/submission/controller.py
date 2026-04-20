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

        ################################### code estelle ##################################
        self.search_step = 0
        self.SPIRAL_SPEED = 0.0001  # tuned, how fast the spiral expands
        '''
        # Searching for odor source (8-figure) inspired from week 3
        turning_speed = 0.5  
        n_steps_full_turn = round(2 * np.pi / np.abs(turning_speed) / sim.timestep)
        turn_right = np.array([1.2, 0.2])
        turn_left = np.array([0.2, 1.2])
        self.search_pattern = np.concatenate([
            np.tile(turn_right, (n_steps_full_turn, 1)),
            np.tile(turn_left, (n_steps_full_turn, 1)),
        ])
        self.search_step = 0  # current position in figure-8
        '''
        # State machine
        self.state = "SEARCH"  # start searching until odor is found
        self.confirm_counter = 0
        #self.CONFIRM_THRESHOLD = 3  # steps of signal needed before switching to FOLLOW
        ###############################################################################


    def step(self, sim: MiniprojectSimulation):
        # implement your control algorithm here
        olfaction = sim.get_olfaction(sim.fly.name)
       
        #print(f"Olfaction: {olfaction}")  # for debugging purposes
        # get other observations as needed
        drives = np.array([2.0, 2.0])  # replace with your control logic
        
        ################################### test joh ##################################

        if self.odor_smooth is None:
            self.odor_smooth = olfaction
        else:
            self.odor_smooth = (1 - self.alpha) * self.odor_smooth + self.alpha * olfaction


        #drives = _odor_to_drives(self.odor_smooth)
                
        ###############################################################################
        ################################### code estelle ##################################
        # TEMPORARY TESTA
        #odor_strength = self.odor_smooth[:, 0].mean()
        odor_strength = 0.0
        # State transitions
        if self.state == "FOLLOW":
            if odor_strength < 1e-7:  # lost the signal
                self.state = "SEARCH"
                self.confirm_counter = 0
                print('lost the smell')
                # don't reset search_step — continue figure-8 where we left off
        elif self.state == "SEARCH":
            if odor_strength > 1e-7:
                self.confirm_counter += 1
                self.state = "FOLLOW"
                print('found the smell')
            else:
                self.confirm_counter = 0  # reset if signal disappears again
        
        # Compute drives based on state
        if self.state == "FOLLOW":
            #print('following mode')
            drives = _odor_to_drives(self.odor_smooth)
        else:  # SEARCH
            #print('searching moode')
            #drives = self.search_pattern[self.search_step % len(self.search_pattern)]
            t = self.search_step
            # start nearly straight, gradually increase turn bias
            bias = max(1 - (t/6) * self.SPIRAL_SPEED, 0.3)  # decrease from 1 to 0.4
            drives = np.array([1.0 + bias, 1.0 - bias])
            self.search_step += 1
        
        ###############################################################################
       
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion


def _odor_to_drives(odor_intensities, attractive_gain=-500, aversive_gain=80):
    # odor_intensities shape : (4, n_sources)
    # n_sources peut être 1 (attractive seulement) ou 2 (attractive + aversive)
    n_sources = odor_intensities.shape[1]

    # Source attractive (dimension 0 — toujours présente)
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