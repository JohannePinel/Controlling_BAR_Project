import numpy as np
from miniproject.simulation import MiniprojectSimulation

# colors
RED = 0
GREEN = 1
BLUE = 2
COLOR_RED = [255, 0, 0]
COLOR_GREEN = [0, 255, 0]
COLOR_BLUE = [0, 0, 255]
COLOR_BLACK = [0, 0, 0]

# rates
VISION_RATE = 200
PITCH_DETECTION_RATE = 50

# ROIs
Y_LOW = 350
Y_HIGH = 250 # In the color vision, the y axis point toward the bottom
X_LEFT = 140
X_RIGHT = 750
MIDDLE_LEFT = 449
MIDDLE_RIGHT = 451

# walking
TURN_RIGHT = 1.25
TURN_LEFT = 1/TURN_RIGHT
CONFIDENT = 1.0
SUSPICIOUS = 0.5
TURN_COEFF = 2.5
FLIPPED_FOR_SURE = 200

class Rectangle:
    def __init__(self, center_x, center_y, width, height, color):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.color = color

    @property
    def bounds(self):
        """Returns the coordinates of the rectangle as (x_min, y_min, x_max, y_max)."""
        x_min = self.center_x - self.width / 2
        y_min = self.center_y - self.height / 2
        x_max = self.center_x + self.width / 2
        y_max = self.center_y + self.height / 2
        return x_min, y_min, x_max, y_max

class ROI:
    def __init__(self, name, y, x0, x1):
        self.name = name
        self.base_y = y
        self.y = y
        self.x0 = x0
        self.x1 = x1
        self.is_active = False
        self.inner_segments = [] # list of (x_start ; x_end) of a segment rpzenting the inner part of an obstacle
        self.edge_indices = []
        self.diff_width = 4


class Controller:
    def __init__(self, sim: MiniprojectSimulation, threshold_line, mode="normal", pitch_weight=1):
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)

        # general parameters
        self.speed = SUSPICIOUS
        self.count = 0
        self.mode = mode
        self.draw_edges = True
        
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

        # Consolidated ROI organization: 1 ROI per height spanning full width
        self.all_rois = [
            ROI("high", Y_HIGH, X_LEFT, X_RIGHT),
            ROI("mid", (Y_HIGH + Y_LOW)//2, X_LEFT, X_RIGHT),
            ROI("low", Y_LOW, X_LEFT, X_RIGHT),
        ]
        self.last_vertical_segments = []        

        # walking inhibition
        self.k = 1

    def step(self, sim: MiniprojectSimulation):
        self.count += 1
                
        # Slope detection
        if self.count % PITCH_DETECTION_RATE == 1: # if was 0, the pitch or the derivative pitch would be reset to 0 right before the color_vision() starts
            self.detect_slope_proprioceptive(sim)
            #self.compute_convexe_concave()

            #if self.current_slope_category == "carreful_upsidedown":
            #    print("!! Retournement imminent !!")

        # Color Vision   
        if self.count % VISION_RATE == 0:
            im = self.color_vision(sim)

            if self.is_flipped(sim, im):
                self.recover_fly(im)

            self.adapts_ROI_to_slope() 
            
            if self.mode == "tuning ROI" or self.mode == "tuning slope":
                self.show_ROI(sim, self.frames[-1])

            else:
                self.detect_line_jump(sim) # contains show_ROI
                self.show_rectangles(im)

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

    def set_edge_drawing(self, status):
        """Toggle the visibility of detected edges (red points) in the visualization."""
        self.draw_edges = status
    
    def show_rectangles(self, im):
        im = self.frames[-1]

    def show_ROI(self, sim: MiniprojectSimulation, im, default_color=COLOR_BLACK):
        for roi in self.all_rois:
            y_draw = int(np.clip(roi.y, 0, im.shape[0]-1))
            im[y_draw, roi.x0:roi.x1] = default_color
            if roi.is_active:
                if self.draw_edges:
                    for i in roi.edge_indices:
                        im[y_draw, (int(i-roi.diff_width/2)):(int(i+roi.diff_width/2))] = COLOR_RED
                
                # Draw inner parts in GREEN
                for start, end in roi.inner_segments:
                    im[y_draw, start:end] = COLOR_GREEN

        # Draw the detected vertical obstacle heights in BLUE (drawn last and thicker for visibility)
        for vx, vy0, vy1 in self.last_vertical_segments:
            vx_start = max(0, vx - 1)
            vx_end = min(im.shape[1], vx + 2)
            im[vy0:vy1, vx_start:vx_end] = COLOR_BLUE

        return im
    
    def detect_line_jump(self, sim: MiniprojectSimulation):
        im = self.frames[-1]
        
        results = []
        for roi in self.all_rois:
            detected = self.detect_line_jump_ROI(im, roi)
            results.append(detected)

        # Vertical height detection based on the most central obstacles per side/ROI
        target_points = self.starting_point()
        self.last_vertical_segments = []
        for vx, vy_start in target_points:
            self.last_vertical_segments.append(self.detect_vertical_obstacle_bounds(im, vx, vy_start))

        self.show_ROI(sim, im)
        return tuple(results)

    def detect_line_jump_ROI(self, im, roi):
        """
        Detects multiple edges and evaluates every interval (including those 
        at the start and end of the scanline) for stability and intensity.
        This allows edges to have inner obstacle parts on both left and right.
        """
        y, x0, x1 = int(roi.y), roi.x0, roi.x1
        green_line = im[y, x0:x1, GREEN].astype(int)
        
        roi.edge_indices = []
        roi.inner_segments = []
        
        if len(green_line) < 10:
            return False

        # 1. Find all edge indices using the threshold
        i = 0
        while i < len(green_line) - roi.diff_width:
            if self.different_intensities(green_line[i + roi.diff_width], green_line[i], self.th_line):
                # Edge detected
                roi.edge_indices.append(i)
                """center = i + diff_width // 2
                # Store this segment
                roi.detected_segments.append((
                    x0 + max(0, center - 5),
                    x0 + min(len(green_line), center + 5)
                ))"""
                # Skip ahead to find the next discrete edge
                i += (roi.diff_width*2)
            else:
                i += 1

        # 2. Analyze all intervals (start of line, edges, end of line) for 'inner' parts.
        # Each edge boundary allows checking for an inner part to its left and right.
        boundary_points = [0] + [idx + 2 for idx in roi.edge_indices] + [len(green_line)]
        
        for k in range(len(boundary_points) - 1):
            idx_start = boundary_points[k]
            idx_end = boundary_points[k+1]
            
            # Avoid checking the pixels immediately inside the edge transition
            check_start = idx_start + (5 if k > 0 else 0)
            check_end = idx_end - (5 if k < len(boundary_points) - 2 else 0)
            
            if check_end > check_start + 2:
                segment = green_line[check_start:check_end]
                blue_segment = im[y, check_start:check_end, BLUE].astype(int)

                # Check for: not in the sky AND not in the grass AND "stable" intensity 
                if not self.in_the_sky(np.mean(blue_segment)) and not self.in_the_grass(np.mean(segment)) and np.ptp(segment) <= 3:
                    roi.inner_segments.append((x0 + idx_start, x0 + idx_end))

        roi.is_active = len(roi.edge_indices) > 0
        return roi.is_active

    def capture_mean_grass(self):
        im = self.frames[-1]
        mean = 0
        line = np.concatenate([im[400, 300:400, GREEN], im[400, 500:600, GREEN]]).astype(int)
        mean = np.mean(line)

        return mean
    
    def different_intensities(self, int1, int2, epsilon):
        if abs(int1 - int2) > epsilon:
            return True
        return False
    
    def in_the_sky(self, value):
        if value > 10:
            return True
        return False
    
    def in_the_grass(self, value):
        if self.different_intensities(value, self.capture_mean_grass(), 20):
            return False
        return True

    def in_the_black(self, point): 
       return all(v < 30 for v in point)

    def detect_vertical_obstacle_bounds(self, im, x, y_start):
        """
        Scans vertically from a seed point to find the upper and lower edges
        of an obstacle using intensity gradients.
        """
        th = self.th_line
        diff = 4
        y_top = 0
        y_bottom = im.shape[0] - 1
        vx = int(np.clip(x, 0, im.shape[1] - 1))
        
        green_v = im[:, vx, GREEN].astype(int)
        blue_v = im[:, vx, BLUE].astype(int)
        red_v = im[:, vx, RED].astype(int)
        
        # Upward scan for upper bound
        for y in range(int(y_start), diff, -1): #because the y-axis is "inverted" so the positive direction is toward the bottom
            
            if self.different_intensities(green_v[y - diff], green_v[y], th) or \
               self.in_the_sky(blue_v[y]) or \
               self.in_the_black([red_v[y], green_v[y], blue_v[y]]):
                y_top = y
                break
        
        # Downward scan for lower bound
        for y in range(int(y_start), im.shape[0] - diff):
            if abs(green_v[y + diff] - green_v[y]) > th:
                y_bottom = y
                break
                
        return (vx, y_top, y_bottom)

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
    
    def starting_point(self):
        """
        Identifies the x-coordinates of the midpoints of the 'inner obstacle' segments
        that are closest to the center for each ROI and each side (left/right).
        
        Returns:
            list: List of (x, y) coordinates of the seed points.
        """
        center_vision_x = (MIDDLE_LEFT + MIDDLE_RIGHT) // 2
        points = []

        for roi in self.all_rois:
            best_left = None
            best_right = None
            min_dist_left = float('inf')
            min_dist_right = float('inf')

            for start, end in roi.inner_segments:
                segment_mid_x = (start + end) / 2
                distance = abs(segment_mid_x - center_vision_x)
                
                if self.is_left(segment_mid_x):
                    if distance < min_dist_left:
                        min_dist_left = distance
                        best_left = (segment_mid_x, roi.y)
                else:
                    if distance < min_dist_right:
                        min_dist_right = distance
                        best_right = (segment_mid_x, roi.y)
            
            if best_left: points.append(best_left)
            if best_right: points.append(best_right)
            
        return points

    def is_left(self, x):
        return x < (MIDDLE_LEFT + MIDDLE_RIGHT) // 2

    def reflexion_obstacle(self):
        for roi in self.all_rois:
            if roi.is_active:
                for start, end in roi.detected_segments:
                    # If segment is on the left half of the vision, turn right
                    if self.is_left((start + end) / 2):
                        self.turn_right()
                        return 0
                    else:
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