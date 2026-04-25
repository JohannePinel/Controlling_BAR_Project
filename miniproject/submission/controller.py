import numpy as np
from miniproject.simulation import MiniprojectSimulation

GREEN = 1
RED = 0

COLOR_RED = [255, 0, 0]
COLOR_GREEN = [0, 255, 0]
COLOR_BLUE = [0, 0, 255]
COLOR_BLACK = [0, 0, 0]

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


class Controller:
    def __init__(self, sim: MiniprojectSimulation, threshold_obstacle, mode="normal", pitch_weight=410):
        # you may also implement your own turning controller
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)

        self.speed = CONFIDENT
        
        # var pr la vision
        self.frames = []
        
        # param détection de pente proprioceptive
        self.current_slope_category = "flat"
        self.pitch = 0
        self.pitch_weight = pitch_weight
        self.th_danger_upsidedown = 0.45
        self.th_slope_category = 0.08

        # param obstacles
        self.count = 0
        self.th_line = 50
        self.window = 16

        # param ROI lignes horizontales (divisées en deux chacune)
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

        # ancien param ROI
        self.al = -0.22
        self.ar = 0.22
        # we will need the initial values of these parameters later. That's why we don't assign them directly a value 
        self.bl = B_LEFT
        self.br = B_RIGHT
        self.bf = B_FRONT
        self.vertml = 400

        # mode 
        self.mode = mode

        # inhibitateur de marche
        self.k = 1

        # var ommatidia
        self.ratio = [0, 0, 0]


    def step(self, sim: MiniprojectSimulation):
        self.count += 1

                
        if self.count % 50 == 1: 
            if self.current_slope_category == "carreful_upsidedown":
                print("!! Retournement imminent !!")
            
        if self.count % 200 == 0:
            self.color_vision(sim)
            self.pitch = self.detect_slope_proprioceptive(sim)
            self.adapts_ROI_to_slope() 
            
            if self.mode == "tuning ROI" or self.mode == "tuning slope":
                self.show_ROI(sim, self.frames[-1], left1=False, left2=False, front_left1=False, front_left2=False, front_left3=False, front_right1=False, front_right2=False, front_right3=False, right1=False, right2=False, fullfill=True)

            else:
                self.detect_line_jump(sim)

        drives = np.array([self.speed*self.k, self.speed/self.k])  
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion

###########################################################
#################### TEST ADRI ############################
###########################################################
    
    def color_vision(self, sim: MiniprojectSimulation):

        vision_data = sim.get_raw_vision(sim.fly.name)
        im = np.concatenate([vision_data[0], vision_data[1]], axis=1)
        self.frames.append(im)

        return 0

    class ROI:
        def __init__(self, color_left_segment, color_right_segment):
            self.color_left_segment = color_left_segment
            self.color_right_segment = color_right_segment
    
    def show_ROI(self, sim: MiniprojectSimulation, im, left1, left2, front_left1, front_left2, front_left3, front_right1, front_right2, front_right3, right1, right2, fullfill=False):
        # left lines
        color_left1 = COLOR_RED if left1 else COLOR_BLACK
        for x in range(self.line_left1_x0, self.line_left1_x1):
            im[self.line_y_left, x] = color_left1

        color_left2 = COLOR_RED if left2 else COLOR_BLACK
        for x in range(self.line_left2_x0, self.line_left2_x1):
            im[self.line_y_left, x] = color_left2


        # front-left lines
        color_front_left1 = COLOR_RED if front_left1 else COLOR_BLACK
        for x in range(self.line_front_left1_x0, self.line_front_left1_x1):
            im[self.line_y_front_left, x] = color_front_left1

        color_front_left2 = COLOR_RED if front_left2 else COLOR_BLACK
        for x in range(self.line_front_left2_x0, self.line_front_left2_x1):
            im[self.line_y_front_left, x] = color_front_left2
        
        color_front_left3 = COLOR_RED if front_left3 else COLOR_BLACK 
        for x in range(self.line_front_left3_x0, self.line_front_left3_x1):
            im[self.line_y_front_left, x] = color_front_left3


        # front-right lines
        color_front_right1 = COLOR_RED if front_right1 else COLOR_BLACK
        for x in range(self.line_front_right1_x0, self.line_front_right1_x1):
            im[self.line_y_front_right, x] = color_front_right1

        color_front_right2 = COLOR_RED if front_right2 else COLOR_BLACK
        for x in range(self.line_front_right2_x0, self.line_front_right2_x1):
            im[self.line_y_front_right, x] = color_front_right2
            
        color_front_right3 = COLOR_RED if front_right3 else COLOR_BLACK
        for x in range(self.line_front_right3_x0, self.line_front_right3_x1):
            im[self.line_y_front_right, x] = color_front_right3


        # right lines
        color_right1 = COLOR_RED if right1 else COLOR_BLACK
        for x in range(self.line_right1_x0, self.line_right1_x1):
            im[self.line_y_right, x] = color_right1

        color_right2 = COLOR_RED if right2 else COLOR_BLACK
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
            right2=detected_right2,
            fullfill=True,
        )

        return detected_left1, detected_left2, detected_front_left1, detected_front_left2, detected_front_left3, detected_front_right1, detected_front_right2, detected_front_right3, detected_right1, detected_right2

    def detect_line_jump_ROI(self, im, y, x0, x1, id_ROI):
        green_line = im[y, x0:x1, GREEN].astype(int)
        red_line = im[y, x0:x1, RED].astype(int)
        
        # If any pixel has non-zero red channel it's that the foot is in the ROI
        if np.any(red_line > 60):
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
                self.ROIs[id_ROI] = (int(left_mean), int(right_mean), True)
                self.reflexion_obstacle()
                return True
            else :
                self.ROIs[id_ROI] = (0, 0, False)

        return False
    
    def detect_slope_proprioceptive(self, sim: MiniprojectSimulation):
        from scipy.spatial.transform import Rotation
        
        # Obtenir l'orientation du thorax (premier corps = index 0)
        body_rotations = sim.get_body_rotations(sim.fly.name)
        thorax_quat = body_rotations[0]  # quaternion du thorax
        
        # Convertir quaternion en angles d'Euler (YAW, PITCH, ROLL)
        # MuJoCo utilise l'ordre ZYX par défaut
        rotation = Rotation.from_quat(thorax_quat)
        euler_angles = rotation.as_euler('xyz')  # [roll, pitch, yaw]
        pitch = euler_angles[1]  # pitch ~[-0.5, 0.5]
        
        if np.abs(pitch) > self.th_danger_upsidedown :
            self.current_slope_category == "carreful_upsidedown"
        else:
            if pitch > self.th_slope_category:
                self.current_slope_category  = "ascending"
            elif pitch < -self.th_slope_category:
                self.current_slope_category  = "descending"
            else:
                self.current_slope_category = "flat"
        
        return pitch
    
    def adapts_ROI_to_slope(self):

        self.line_y_left = int(Y_RIGHT + self.pitch*self.pitch_weight)
        self.line_y_front_left = int(Y_FRONTL + self.pitch*self.pitch_weight)
        self.line_y_front_right = int(Y_FRONTR + self.pitch*self.pitch_weight)
        self.line_y_right = int(Y_LEFT + self.pitch*self.pitch_weight)

        return 0
    
    def reflexion_obstacle(self):

        for i in [0, 1, 8, 9]: # checks first the bottom ROIs
            if self.speed == SUSPICIOUS and self.ROIs[i][2]:
                if i < 2 : # obstacle is at left
                    self.k = 2 # turn right
                else :
                    self.k = (1/2) # turn left
            else : 
                self.k = 1 # no turn yet

        for i in  [2, 3, 4, 5, 6, 7]:
            if self.ROIs[i][2]: # if this ROI is active
                self.speed = SUSPICIOUS

        return 0

    def compute_convexe_concave(self):

        return 0
    