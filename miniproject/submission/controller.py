import numpy as np
from miniproject.simulation import MiniprojectSimulation

from scipy.spatial.transform import Rotation
from flygym.examples.locomotion import TurningController


RED = 0
GREEN = 1
BLUE = 2

COLOR_RED = [255, 0, 0]
COLOR_GREEN = [0, 255, 0]
COLOR_BLUE = [0, 0, 255]
COLOR_BLACK = [0, 0, 0]

VISION_RATE = 200
PITCH_DETECTION_RATE = 50

TURN_RIGHT = 1.25
TURN_LEFT = 1/TURN_RIGHT

B_LEFT = 435
B_RIGHT = 240
B_FRONT = 320
TH_FRONT = 20

Y_LEFT = 360
Y_FRONTL = 320
Y_FRONTR = 320
Y_RIGHT = 360

CONFIDENT = 1.0
SUSPICIOUS = 0.4 # Johanne, j'augmente par ce que c'est lent là
TURN_COEFF = 2.5
FLIPPED_FOR_SURE = 200

TURN_DURATION = 100  # La mouche tourne pendant ~100 steps puis réévalue


class ROI:
    def __init__(self, name, side, y, x0, x1):
        self.name = name
        self.side = side  # 'left', 'front_left', 'front_right', 'right'
        self.base_y = y
        self.y = y
        self.x0 = x0
        self.x1 = x1
        self.is_active = False
        self.intensity = (0, 0)

class Controller:
    def __init__(self, sim: MiniprojectSimulation, threshold_line = 40, mode="normal", pitch_weight=1): 
        self.turning_controller = TurningController(sim.timestep)

        # general parameters
        self.speed = SUSPICIOUS
        self.count = 0
        self.mode = mode
        
        # color vision parameters
        self.frames = []
        
        # param détection de pente proprioceptive
        self.current_slope_category = "flat"
        self.pitch = 0
        self.prev_pitch = 5 # if it was 0, the first computation of the pitch_derivative is too high and the function adapts_ROI_to_slope wants to put the ROI to a height which is above the limit of the frame (512x900)
        self.pitch_derivative = 0
        self.pitch_count = 1
        self.pitch_collection_window = 4 
        self.pitch_weight = pitch_weight
        self.th_danger_upsidedown = 0.48
        self.th_slope_category = 0.08
        self.maybe_flipped = 0

        #johanne code
        self.odor_smooth = None
        self.alpha = 0.1  # smoothing factor for the low-pass filter
                
        # param obstacles
        self.th_line = threshold_line
        self.window = 16

        # Centralized ROI organization
        self.all_rois = [
            ROI("left1", "left", Y_RIGHT, 220, 300), ROI("left2", "left", Y_RIGHT, 300, 380),
            ROI("front_left1", "front_left", Y_FRONTL, 300, 370), ROI("front_left2", "front_left", Y_FRONTL, 335, 405), ROI("front_left3", "front_left", Y_FRONTL, 370, 440),
            ROI("front_right1", "front_right", Y_FRONTR, 460, 530), ROI("front_right2", "front_right", Y_FRONTR, 495, 565), ROI("front_right3", "front_right", Y_FRONTR, 530, 600),
            ROI("right1", "right", Y_LEFT, 520, 600), ROI("right2", "right", Y_LEFT, 600, 680),
        ]

        # walking inhibition
        self.k = 1
        self.turn_timer = 0
        self.turning = False  # est-ce qu'on est en train de virer ?

    def step(self, sim: MiniprojectSimulation):
        self.count += 1

        ########## odor to drive ##########

        olfaction = sim.get_olfaction(sim.fly.name)

        
        if self.odor_smooth is None:
            self.odor_smooth = olfaction
        else:
            self.odor_smooth = (1 - self.alpha) * self.odor_smooth + self.alpha * olfaction

        odor_drives = odor_to_drives(self.odor_smooth) * 2.5 #fois x to increase the effect of the odor on the speed, otherwise the fly is too much focused on the obstacle avoidance and doesn't move enough towards the target
        ############################################################
                
        # Slope detection
        if self.count % PITCH_DETECTION_RATE == 1: # if was 0, the pitch or the derivative pitch would be reset to 0 right before the color_vision() starts
            self.detect_slope_proprioceptive(sim)
            self.compute_convexe_concave()

            if self.current_slope_category == "carreful_upsidedown":
                print("!! Retournement imminent !!")

        # Color Vision   
        if self.count % VISION_RATE == 0:
            im = self.color_vision(sim)

            if self.is_flipped(sim, im):
                self.recover_fly(im)

            self.adapts_ROI_to_slope() 
            
            if self.mode == "tuning ROI" or self.mode == "tuning slope":
                self.show_ROI(sim, self.frames[-1])

            else: # Obstacle detection
                self.detect_line_jump(sim)
                self.reflexion_obstacle()
        
        self.update_turn_timer()

        drives = self.speed * odor_drives * np.array([self.k, 1/self.k])


        #drives = np.array([self.speed*self.k, self.speed/self.k])  
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion

##########################################################################################################



##########################################################################################################
##########################################################################################################
    
    def color_vision(self, sim: MiniprojectSimulation):

        vision_data = sim.get_raw_vision(sim.fly.name)
        im = np.concatenate([vision_data[0], vision_data[1]], axis=1)
        self.frames.append(im)

        return im

    def show_ROI(self, sim: MiniprojectSimulation, im, default_color=COLOR_BLACK):
        for roi in self.all_rois:
            color = COLOR_RED if roi.is_active else default_color
            # Ensure y is within image bounds for drawing
            y_draw = int(np.clip(roi.y, 0, im.shape[0]-1))
            im[y_draw, roi.x0:roi.x1] = color
        return im
    
    def detect_line_jump(self, sim: MiniprojectSimulation):
        im = self.frames[-1]
        
        results = []
        for roi in self.all_rois:
            detected = self.detect_line_jump_ROI(im, roi)
            results.append(detected)

        self.show_ROI(sim, im)
        return tuple(results)

    def detect_line_jump_ROI(self, im, roi):
        y, x0, x1 = int(roi.y), roi.x0, roi.x1
        green_line = im[y, x0:x1, GREEN].astype(int)
        red_line = im[y, x0:x1, RED].astype(int)
        
        if np.any(red_line > 60): # pixel with too much red might be the fly's foot
            return False
            
        if green_line.shape[0] < 10:  # need enough pixels to check stability on both sides
            return False

        for i in range(len(green_line) - 3):
            left_value = green_line[i]
            right_value = green_line[i + 3]
            if abs(right_value - left_value) <= self.th_line: # a change too suden of green intensity
                continue

            left_start = max(0, i - self.window)
            right_end = min(len(green_line), i + 3 + self.window)

            left_segment = green_line[left_start:i + 1]
            right_segment = green_line[i + 3:right_end]

            if len(left_segment) < 3 or len(right_segment) < 3:
                continue

            left_std = np.std(left_segment)
            right_std = np.std(right_segment)
            left_mean = np.mean(left_segment)
            right_mean = np.mean(right_segment)

            if left_std < 10 and right_std < 10 and abs(left_mean - right_mean) > self.th_line:
                roi.intensity = (int(left_mean), int(right_mean))
                roi.is_active = True
                return True
        
        roi.is_active = False
        return False
    
    def detect_slope_proprioceptive(self, sim: MiniprojectSimulation):
        
        # Obtenir l'orientation du thorax (premier corps = index 0)
        body_rotations = sim.get_body_rotations(sim.fly.name)
        # Reorder MuJoCo [w, x, y, z] to SciPy [x, y, z, w]
        thorax_quat = body_rotations[0][[1, 2, 3, 0]]
        
        # Convertir quaternion en angles d'Euler
        rotation = Rotation.from_quat(thorax_quat)
        euler_angles = rotation.as_euler('xyz')  # [roll, pitch, yaw]
        pitch = euler_angles[1]  # pitch ~[-0.5, 0.5]
        
        if np.abs(pitch) > self.th_danger_upsidedown :
            self.current_slope_category = "carreful_upsidedown"
        else:
            if pitch > self.th_slope_category:
                self.current_slope_category  = "ascending"
            elif pitch < -self.th_slope_category:
                self.current_slope_category  = "descending"
            else:
                self.current_slope_category = "flat"

            self.prev_pitch = self.pitch
            self.pitch = pitch
        
        return 0
    
    def adapts_ROI_to_slope(self): # when derivative is positive, ROI need to be little higher (and when negative, needs to be a little lower)
        coeff = self.pitch_derivative if self.mode == "adapts_ROI derivative" else self.pitch
        coeff *= self.pitch_weight

        for roi in self.all_rois:
            new_y = roi.base_y + coeff
            roi.y = np.clip(new_y, 75, 475)
            
        self.pitch_derivative = 0
        return 0
    
    def reflexion_obstacle(self):
        for roi in self.all_rois:
            if roi.is_active:
                #print(f"Obstacle detected in {roi.name} with intensity {roi.intensity}")
                if "left" in roi.side:
                    self.turn_right()
                    return 0
                elif "right" in roi.side:
                    self.turn_left()
                    return 0

        self.no_turn() # No obstacles detected in any ROI
        return 0

    def turn_right(self): 
        self.k = TURN_COEFF
        print("turning right")
        self.turning = True
        self.turn_timer = TURN_DURATION

    def turn_left(self): 
        self.k = 1/TURN_COEFF
        self.turning = True
        self.turn_timer = TURN_DURATION

    def no_turn(self): 
        self.k = 1
        self.turning = False
        self.turn_timer = 0

    def update_turn_timer(self):
        if self.turning and self.turn_timer > 0:
            self.turn_timer -= 1
            if self.turn_timer == 0:
                self.no_turn()


    def compute_convexe_concave(self):
        window = (VISION_RATE/PITCH_DETECTION_RATE+1)
        if self.pitch_count < window:
            self.pitch_derivative += ((self.prev_pitch - self.pitch)/self.speed)
            self.pitch_count += 1
        if self.pitch_count == window : self.pitch_count = 1
        return 0

    def is_flipped(self, sim: MiniprojectSimulation, im):
        for roi in self.all_rois:
            blue_line = im[int(roi.y), roi.x0:roi.x1, BLUE].astype(int)
            if np.any(blue_line > 0):
                if self.maybe_flipped >= FLIPPED_FOR_SURE:   
                    self.maybe_flipped = 0
                    return True
                else:   
                    self.maybe_flipped += 1
        return False

    def recover_fly(self, im):
        im[110:160, 110:160] = COLOR_BLACK
        return 0
    
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