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
        ################################### code djo ##################################
        self.vision_frames = []  # frames lisibles pour affichage
        self.step_count = 0
        ###############################################################################
       

    def step(self, sim: MiniprojectSimulation):
        # implement your control algorithm here
        self.step_count += 1

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

        ########################## vision obstacle avoidance #################
        self.ommatidia = sim.get_ommatidia_readouts(sim.fly.name)

        self.left_intensity = self.ommatidia[0].mean()
        self.right_intensity = self.ommatidia[1].mean()

        if self.step_count % 100 == 0:
            im = np.concatenate([
                sim.fly.retina.hex_pxls_to_human_readable(eye.max(-1), color_8bit=True)
                for eye in self.ommatidia
            ], axis=1)
            self.vision_frames.append(im)
            
        obstacle_left = 1 - self.left_intensity 
        obstacle_right = 1 - self.right_intensity
        obstacle_max = max(obstacle_left, obstacle_right)
        ###############################################################################
                
        ###############################################################################
        ################################### code estelle ##################################

        odor_strength = self.odor_smooth[:, 0].mean()
    
        # State transitions
        obstacle_threshold = 2    # à tester
        if obstacle_max > obstacle_threshold:
            print("we are avoiding")
            self.state = "AVOID"

        if self.state == "AVOID":
            if obstacle_max < obstacle_threshold:
                self.state =="SEARCH"
                print("finished avoiding, back to search")


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
        if self.state == "AVOID":
            if obstacle_left > obstacle_right:
                drives[0] -= 0.3 # tourner à droite
            else:
                drives[1] -= 0.3  # tourner à gauche
        elif self.state == "FOLLOW":
            #print('following mode')
            drives = odor_to_drives(self.odor_smooth)
        elif self.state == "SEARCH":
            #print('searching moode')
            #drives = self.search_pattern[self.search_step % len(self.search_pattern)]
            t = self.search_step
            # start nearly straight, gradually increase turn bias
            bias = max(0.8 - (t/6) * self.SPIRAL_SPEED, 0.3)  # decrease from 1 to 0.4
            drives = np.array([1.0 + bias, 1.0 - bias])
            self.search_step += 1
        else :
            print("lost in between states")
        
        ###############################################################################


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