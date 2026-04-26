import numpy as np
from miniproject.simulation import MiniprojectSimulation


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
SUSPICIOUS = 0.5
TURN_COEFF = 4
FLIPPED_FOR_SURE = 200

class Controller:
    def __init__(self, sim: MiniprojectSimulation, threshold_line = 40, mode="normal", pitch_weight=1):
        from flygym.examples.locomotion import TurningController
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
        self.th_danger_upsidedown = 0.45
        self.th_slope_category = 0.08
        self.maybe_flipped = 0

        # param obstacles
        self.th_line = threshold_line
        self.window = 16

        # ROI parameters
        self.ROIs = [] # (intensity_left_segment, intensity_right_segment, status(ON/OFF))
        self.line_y_left = Y_RIGHT
        self.line_left1_x0 = 220
        self.line_left1_x1 = 300
        self.ROIs.append((0, 0, False)) # False meaning "this ROI is not detecting anything particular"
        self.line_left2_x0 = 300
        self.line_left2_x1 = 380
        self.ROIs.append((0, 0, False))

        self.line_y_front_left = Y_FRONTL
        self.line_front_left1_x0 = 300
        self.line_front_left1_x1 = 370
        self.ROIs.append((0, 0, False)) 
        self.line_front_left2_x0 = 335
        self.line_front_left2_x1 = 405
        self.ROIs.append((0, 0, False))
        self.line_front_left3_x0 = 370
        self.line_front_left3_x1 = 440
        self.ROIs.append((0, 0, False))

        self.line_y_front_right = Y_FRONTR
        self.line_front_right1_x0 = 460
        self.line_front_right1_x1 = 530
        self.ROIs.append((0, 0, False))
        self.line_front_right2_x0 = 495
        self.line_front_right2_x1 = 565
        self.ROIs.append((0, 0, False))
        self.line_front_right3_x0 = 530
        self.line_front_right3_x1 = 600
        self.ROIs.append((0, 0, False))

        self.line_y_right = Y_LEFT
        self.line_right1_x0 = 520
        self.line_right1_x1 = 600
        self.ROIs.append((0, 0, False))
        self.line_right2_x0 = 600
        self.line_right2_x1 = 680
        self.ROIs.append((0, 0, False))

        # walking inhibition
        self.k = 1

        # var ommatidia
        self.ratio = [0, 0, 0]

    def _set_line_y(self, attr_name, value):
        
        if value < 0:  
            setattr(self, attr_name, 75)
        elif value > 512:  
            setattr(self, attr_name, 450)
        else:
            setattr(self, attr_name, value)


    def step(self, sim: MiniprojectSimulation):
        self.count += 1
                
        if self.count % PITCH_DETECTION_RATE == 1: # if was 0, the pitch or the derivative pitch would be reset to 0 right before the color_vision() starts
            self.detect_slope_proprioceptive(sim)
            self.compute_convexe_concave()

            if self.current_slope_category == "carreful_upsidedown":
                print("!! Retournement imminent !!")
            
        if self.count % VISION_RATE == 0:
            im = self.color_vision(sim)

            if self.is_flipped(sim, im):
                self.recover_fly(im)

            self.adapts_ROI_to_slope() 
            
            if self.mode == "tuning ROI" or self.mode == "tuning slope":
                self.show_ROI(sim, self.frames[-1], left1=False, left2=False, front_left1=False, front_left2=False, front_left3=False, front_right1=False, front_right2=False, front_right3=False, right1=False, right2=False, fullfill=True)

            else:
                self.detect_line_jump(sim)
                self.reflexion_obstacle()

        drives = np.array([self.speed*self.k, self.speed/self.k])  
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

    class ROI:
        def __init__(self, color_left_segment, color_right_segment):
            self.color_left_segment = color_left_segment
            self.color_right_segment = color_right_segment
    
    def show_ROI(self, sim: MiniprojectSimulation, im, left1, left2, front_left1, front_left2, front_left3, front_right1, front_right2, front_right3, right1, right2, color=COLOR_BLACK):
        # left lines
        color_left1 = COLOR_RED if left1 else color
        for x in range(self.line_left1_x0, self.line_left1_x1):
            im[self.line_y_left, x] = color_left1

        color_left2 = COLOR_RED if left2 else color
        for x in range(self.line_left2_x0, self.line_left2_x1):
            im[self.line_y_left, x] = color_left2


        # front-left lines
        color_front_left1 = COLOR_RED if front_left1 else color
        for x in range(self.line_front_left1_x0, self.line_front_left1_x1):
            im[self.line_y_front_left, x] = color_front_left1

        color_front_left2 = COLOR_RED if front_left2 else color
        for x in range(self.line_front_left2_x0, self.line_front_left2_x1):
            im[self.line_y_front_left, x] = color_front_left2
        
        color_front_left3 = COLOR_RED if front_left3 else color 
        for x in range(self.line_front_left3_x0, self.line_front_left3_x1):
            im[self.line_y_front_left, x] = color_front_left3


        # front-right lines
        color_front_right1 = COLOR_RED if front_right1 else color
        for x in range(self.line_front_right1_x0, self.line_front_right1_x1):
            im[self.line_y_front_right, x] = color_front_right1

        color_front_right2 = COLOR_RED if front_right2 else color
        for x in range(self.line_front_right2_x0, self.line_front_right2_x1):
            im[self.line_y_front_right, x] = color_front_right2
            
        color_front_right3 = COLOR_RED if front_right3 else color
        for x in range(self.line_front_right3_x0, self.line_front_right3_x1):
            im[self.line_y_front_right, x] = color_front_right3


        # right lines
        color_right1 = COLOR_RED if right1 else color
        for x in range(self.line_right1_x0, self.line_right1_x1):
            im[self.line_y_right, x] = color_right1

        color_right2 = COLOR_RED if right2 else color
        for x in range(self.line_right2_x0, self.line_right2_x1):
            im[self.line_y_right, x] = color_right2

        return im
    
    def detect_line_jump(self, sim: MiniprojectSimulation):
        im = self.frames[-1]

        detected_left1 = self.detect_line_jump_ROI(im, self.line_y_left, self.line_left1_x0, self.line_left1_x1, 0)
        detected_left2 = self.detect_line_jump_ROI(im, self.line_y_left, self.line_left2_x0, self.line_left2_x1, 1)
        detected_front_left1 = self.detect_line_jump_ROI(im, self.line_y_front_left, self.line_front_left1_x0, self.line_front_left1_x1, 2)
        detected_front_left2 = self.detect_line_jump_ROI(im, self.line_y_front_left, self.line_front_left2_x0, self.line_front_left2_x1, 3)
        detected_front_left3 = self.detect_line_jump_ROI(im, self.line_y_front_left, self.line_front_left3_x0, self.line_front_left3_x1, 4)
        detected_front_right1 = self.detect_line_jump_ROI(im, self.line_y_front_right, self.line_front_right1_x0, self.line_front_right1_x1, 5)
        detected_front_right2 = self.detect_line_jump_ROI(im, self.line_y_front_right, self.line_front_right2_x0, self.line_front_right2_x1, 6)
        detected_front_right3 = self.detect_line_jump_ROI(im, self.line_y_front_right, self.line_front_right3_x0, self.line_front_right3_x1, 7)
        detected_right1 = self.detect_line_jump_ROI(im, self.line_y_right, self.line_right1_x0, self.line_right1_x1, 8)
        detected_right2 = self.detect_line_jump_ROI(im, self.line_y_right, self.line_right2_x0, self.line_right2_x1, 9)

        self.show_ROI(
            sim,
            im,
            left1=detected_left1,
            left2=detected_left2,
            front_left1=detected_front_left1,
            front_left2=detected_front_left2,
            front_left3=detected_front_left3,
            front_right1=detected_front_right1,
            front_right2=detected_front_right2,
            front_right3=detected_front_right3,
            right1=detected_right1,
            right2=detected_right2
        )

        return detected_left1, detected_left2, detected_front_left1, detected_front_left2, detected_front_left3, detected_front_right1, detected_front_right2, detected_front_right3, detected_right1, detected_right2

    def detect_line_jump_ROI(self, im, y, x0, x1, id_ROI):
        
        green_line = im[y, x0:x1, GREEN].astype(int)
        red_line = im[y, x0:x1, RED].astype(int)
        
        if np.any(red_line > 60): # pixel with too much red might be the fly's foot
            return False
            
        if green_line.shape[0] < 10:  # need enough pixels to check stability on both sides
            return False

        for i in range(len(green_line) - 3):
            left_value = green_line[i]
            right_value = green_line[i + 3]
            if abs(right_value - left_value) <= self.th_line:
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
                self.ROIs[id_ROI] = (int(left_mean), int(right_mean), True) # this ROI is active <=> obstacle probably there
                self.reflexion_obstacle()
                return True
            else :
                self.ROIs[id_ROI] = (0, 0, False)

        return False
    
    def detect_slope_proprioceptive(self, sim: MiniprojectSimulation):
        from scipy.spatial.transform import Rotation
        
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

        if self.mode == "adapts_ROI derivative":
            coeff = self.pitch_derivative
            #if self.count > 2000 and self.count % VISION_RATE == 0:
                #print(self.pitch_derivative)
        else :
            coeff = self.pitch

        coeff *= self.pitch_weight
        self._set_line_y('line_y_left', int(Y_RIGHT + coeff))
        self._set_line_y('line_y_front_left', int(Y_FRONTL + coeff))
        self._set_line_y('line_y_front_right', int(Y_FRONTR + coeff))
        self._set_line_y('line_y_right', int(Y_LEFT + coeff))
        
        self.pitch_derivative = 0

        return 0

    
    def reflexion_obstacle(self):

        for i in range(len(self.ROIs)): # when coeff == 0 ->checks the left side ROIs, when coeff==1 ->checks the right ROIs

            # Check if the ROI is active (index 2 is the boolean status)
            if self.ROIs[i][2]:
                if i < (len(self.ROIs)/2):
                    self.turn_right()
                    return 0 # Stop checking once we decide to turn
                else:
                    self.turn_left()
                    return 0 # Stop checking once we decide to turn

        self.no_turn() # No obstacles detected in any ROI
        return 0
    
    def turn_right(self):
        self.k = TURN_COEFF
        return 0
    
    def turn_left(self):
        self.k = 1/TURN_COEFF
        return 0
    
    def no_turn(self):
        self.k = 1
        return 0

    def compute_convexe_concave(self): # wokrs better with dt (temporal rather than "geographical")

        window = (VISION_RATE/PITCH_DETECTION_RATE+1) # if pitch computed each 50setps and vision each 200steps, we will need the mean of last 4 derivative computations to compute the average pitch

        if self.pitch_count < window:
            self.pitch_derivative += ((self.prev_pitch - self.pitch)/self.speed) 
            # The sign is inverted as the y in the frames is increasing toward the bottom. So when the pitch is increasing -> I need the ROI to go higher -> I need a line_y to be lower
            # Also, no need to divide by the distance and count, because we can play with the parameter self.pitch_weight. We just need the denominator to be proportional to the speed
            self.pitch_count += 1

        if self.pitch_count == window : # enough data to be sure about the "mean" previous pitch
            self.pitch_count = 1
        
        return 0

    def _get_all_roi_coords(self):
        """Helper to return a list of all ROI segments (y, x0, x1)."""
        return [
            (self.line_y_left, self.line_left1_x0, self.line_left1_x1),
            (self.line_y_left, self.line_left2_x0, self.line_left2_x1),
            (self.line_y_front_left, self.line_front_left1_x0, self.line_front_left1_x1),
            (self.line_y_front_left, self.line_front_left2_x0, self.line_front_left2_x1),
            (self.line_y_front_left, self.line_front_left3_x0, self.line_front_left3_x1),
            (self.line_y_front_right, self.line_front_right1_x0, self.line_front_right1_x1),
            (self.line_y_front_right, self.line_front_right2_x0, self.line_front_right2_x1),
            (self.line_y_front_right, self.line_front_right3_x0, self.line_front_right3_x1),
            (self.line_y_right, self.line_right1_x0, self.line_right1_x1),
            (self.line_y_right, self.line_right2_x0, self.line_right2_x1)
        ]

    def is_flipped(self, sim: MiniprojectSimulation, im): 
        
        for y, x0, x1 in self._get_all_roi_coords():
            blue_line = im[int(y), x0:x1, BLUE].astype(int)
            if np.any(blue_line > 0):
                if self.maybe_flipped >= FLIPPED_FOR_SURE:   
                    self.maybe_flipped = 0
                    return True
                else:   
                    self.maybe_flipped += 1
        return False

    def recover_fly(self, im):
        im[10:60, 10:60] = COLOR_BLACK
        return 0