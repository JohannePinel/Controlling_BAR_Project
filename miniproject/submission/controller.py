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

Y_LOW = 350
Y_HIGH = 250 # In the color vision, the y axis point toward the bottom
MIDDLE_LEFT = 449
MIDDLE_RIGHT = 451

CONFIDENT = 1.0
SUSPICIOUS = 0.5
TURN_COEFF = 2.5
FLIPPED_FOR_SURE = 200

class ROI:
    def __init__(self, name, side, y, x0, x1):
        self.name = name
        self.side = side  # 'left', 'front_left', 'front_right', 'right'
        self.base_y = y
        self.y = y
        self.x0 = x0
        self.x1 = x1
        self.is_active = False
        self.start_x = x0
        self.end_x = x1

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
        self.th_danger_upsidedown = 0.48
        self.th_slope_category = 0.08
        self.maybe_flipped = 0

        # param obstacles
        self.th_line = threshold_line
        self.window = 16

        # Centralized ROI organization
        # 3 ROIs per side, staggered between Y=320 and Y=360
        self.all_rois = [
            ROI("left1", "left", Y_HIGH, 200, MIDDLE_LEFT),
            ROI("left2", "left", (Y_HIGH + Y_LOW)//2, 200, MIDDLE_LEFT),
            ROI("left3", "left", Y_LOW, 200, MIDDLE_LEFT),
            ROI("right1", "right", Y_LOW, MIDDLE_RIGHT, 665),
            ROI("right2", "right", (Y_HIGH + Y_LOW)//2, MIDDLE_RIGHT, 665),
            ROI("right3", "right", Y_HIGH, MIDDLE_RIGHT, 665),
        ]

        # walking inhibition
        self.k = 1

    def step(self, sim: MiniprojectSimulation):
        self.count += 1
                
        # Slope detection
        if self.count % PITCH_DETECTION_RATE == 1: # if was 0, the pitch or the derivative pitch would be reset to 0 right before the color_vision() starts
            self.detect_slope_proprioceptive(sim)
            #self.compute_convexe_concave()

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
                #self.reflexion_obstacle()

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

    def show_ROI(self, sim: MiniprojectSimulation, im, default_color=COLOR_BLACK):
        for roi in self.all_rois:
            y_draw = int(np.clip(roi.y, 0, im.shape[0]-1))
            if roi.is_active:
                # Draw background strip in black and the detected obstacle in RED
                im[y_draw, roi.x0:roi.x1] = COLOR_BLACK
                im[y_draw, roi.start_x:roi.end_x] = COLOR_RED
            else:
                im[y_draw, roi.x0:roi.x1] = default_color
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
        
        # Reverse the scan for left-side ROIs to start from the center of vision
        is_left = (roi.side == "left")
        if is_left:
            green_line = green_line[::-1]
            red_line = red_line[::-1]

        # 1. Locate the first abrupt change (edge)
        idx1 = -1
        for i in range(len(green_line) - 3):
            if abs(green_line[i+3] - green_line[i]) > self.th_line:
                idx1 = i
                break
        
        if idx1 == -1:
            roi.is_active = False
            return False

        # 2. Check stability (tolerance 3) on both sides of the edge to find the obstacle direction
        win = 4 # neighborhood size
        r_stable = False
        if idx1 + 3 + win < len(green_line):
            seg = green_line[idx1 + 3 : idx1 + 3 + win]
            if (np.max(seg) - np.min(seg)) <= 3:
                r_stable = True

        l_stable = False
        if idx1 - win >= 0:
            seg = green_line[idx1 - win : idx1]
            if (np.max(seg) - np.min(seg)) <= 3:
                l_stable = True

        # 3. Search for the second abrupt change in the stable direction
        found = False
        rel_start, rel_end = -1, -1
        if r_stable:
            for j in range(idx1 + 3 + win, len(green_line) - 3):
                if abs(green_line[j+3] - green_line[j]) > self.th_line:
                    rel_start, rel_end = idx1, j + 3
                    found = True
                    break
        
        elif l_stable:
            for j in range(idx1 - win - 3, -1, -1):
                if abs(green_line[j+3] - green_line[j]) > self.th_line:
                    rel_start, rel_end = j, idx1 + 3
                    found = True
                    break

        if found:
            if is_left:
                # Map coordinates back from reversed space to original image space
                line_len = len(green_line)
                roi.start_x = x0 + (line_len - rel_end)
                roi.end_x = x0 + (line_len - rel_start)
            else:
                roi.start_x = x0 + rel_start
                roi.end_x = x0 + rel_end
            roi.is_active = True
            return True

        roi.is_active = False
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
                if "left" in roi.side:
                    self.turn_right()
                    return 0
                elif "right" in roi.side:
                    self.turn_left()
                    return 0

        self.no_turn() # No obstacles detected in any ROI
        return 0

    def turn_right(self): self.k = TURN_COEFF
    def turn_left(self): self.k = 1/TURN_COEFF
    def no_turn(self): self.k = 1

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