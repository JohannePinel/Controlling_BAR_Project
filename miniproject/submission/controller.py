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

# rates
VISION_RATE = 200
PITCH_DETECTION_RATE = 50
OLFACTION_RATE = 100

# ROI
Y_LOW = 350
Y_LOW_DRAG = 450
Y_HIGH = 250 
Y_HIGH_DRAG = 150
Y_ADITIONAL = 200 # In the color vision, the y axis point toward the bottom
X_LEFT = 140
X_LEFT_DRAG = 0
X_RIGHT = 750
X_RIGHT_DRAG = 900
MIDDLE_LEFT = 449
MIDDLE_RIGHT = 451
NB_OF_ROI_DRAG = 22


# rectangles
NB_OF_RECT = 4
WIDTH_INCR = (X_RIGHT - X_LEFT)//NB_OF_RECT
HEIGHT_INCR = (Y_LOW - Y_HIGH)//2

# walking
TURN_RIGHT = 1.25
TURN_LEFT = 1/TURN_RIGHT
CONFIDENT = 1.0
SUSPICIOUS = 0.8
TURN_COEFF = 2.5
FLIPPED_FOR_SURE = 200
RUN_AWAY_FROM_DRAGONFLY = 1.5

AVOIDANCE_DURATION = 200 
DANGER_THRESHOLD = 100 
AVOIDANCE_DRAGONFLY_DURATION = 100

class Rectangle:
    def __init__(self, x0, x1, y_top, y_bottom, color):
        self.x0 = x0
        self.x1 = x1
        self.y_top = y_top
        self.y_bottom = y_bottom
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
        self.inner_segments = [] # list of (x_start ; x_end) 
        self.edge_indices = []
        self.diff_width = 4

class Controller:
    def __init__(self, sim: MiniprojectSimulation, threshold_line = 40, mode="normal", pitch_weight=1): 
        self.turning_controller = TurningController(sim.timestep)

        # ========General parameters========
        self.speed = SUSPICIOUS
        self.count = 0
        self.mode = mode
        self.draw_edges = True
        
        # ========Color vision parameters========
        self.frames = []
        
        # ========Proprioception paramters========
        self.current_slope_category = "flat"
        self.pitch = 0
        self.pitch_derivative = 0
        self.pitch_count = 1
        self.pitch_collection_window = 4 
        self.pitch_weight = pitch_weight
        self.th_danger_upsidedown = 0.3
        self.th_slope_category = 0.08
        self.maybe_flipped = 0
        self.prev_pitch = 5 # if it was 0, the first computation of the pitch_derivative is too high and 
                            # the function adapts_ROI_to_slope wants to put the ROI to a height which is above the limit of the frame (512x900)

        # ========Detection parameters========
        self.th_line = threshold_line
        self.window = 16
        self.last_vertical_segments = []   
        self.all_rois_obst = [ 
                ROI("HIGH++", Y_ADITIONAL, X_LEFT, X_RIGHT),
                ROI("high", Y_HIGH, X_LEFT, X_RIGHT),
                ROI("mid", (Y_HIGH + Y_LOW)//2, X_LEFT, X_RIGHT),
                ROI("low", Y_LOW, X_LEFT, X_RIGHT),
            ]

        self.all_rectangles = []
        for i in range(NB_OF_RECT):
            self.all_rectangles.append(Rectangle(X_LEFT+(i*WIDTH_INCR)-1, X_LEFT+((i+1)*WIDTH_INCR), self.all_rois_obst[0].y - HEIGHT_INCR, self.all_rois_obst[2].y + HEIGHT_INCR, COLOR_BLACK))
                                                # the "-1" is here so that the beginning of a rectangle do not 
                                                # overlapp the end of the previous one  

        # ======== Dragonfly ========
        self.all_rois_drag = [
            ROI(f"drag_{i}", int(y), X_LEFT_DRAG, X_RIGHT_DRAG)
            for i, y in enumerate(np.linspace(Y_HIGH_DRAG, Y_LOW_DRAG, NB_OF_ROI_DRAG))
        ]
        self.dragonfly_pos = (0, 0) # (y, x)

        # ========Johanne code========
        self.odor_smooth = None
        self.alpha = 0.1                        # smoothing factor for the low-pass filter        

        # ======== Obstacle avoidance ========
        self.avoiding_obstacle = False
        self.avoiding_dragonfly = False
        self.avoidance_timer = 0
        self.avoidance_dragonfly_timer = 0
        self.avoidance_direction = 0            # -1 = gauche, 1 = droite, 0 = pas d'évitement
        self.danger_zone_index = -1             # Index de la zone dangereuse (0-3)

        # ======== walking inhibition ========
        self.k = 1
        self.turn_timer = 0
        self.turning = False                    # est-ce qu'on est en train de virer ?
        self.trajectory = []                    # Liste de (x, y) positions de la mouche
        self.obstacle_detection_points = []     # Liste de (x, y) où un obstacle a été détecté
        self.obs = False
        self.odor_drives = np.ones(2)

        # ======== wind analysis ========
        self.wind_state = "NONE"
        self.step_count = 0
        self.t_last_odor = 0          # timestep of last odor detection
        self.last_odor_drives = np.ones(2)
        self.odor_state = "LOST"      # "TRACKING", "RECOVERING", "LOST"
        self.prev_state = "LOST"
        self.LOST_THRESHOLD = 12     # from paper, 25-38 steps
        #self.RECOVERING_THRESHOLD = 12 
        self.odor_memory_fast = 0.0        # drives turn decisions
        self.alpha_fast = 2/(8+1)   # ~0.22
        self.upside_down = False 

###################################################################################
################################# STEP FUNCTION ###################################
###################################################################################
    def step(self, sim: MiniprojectSimulation):
        self.count += 1

        fly_pos = sim.get_body_positions(sim.fly.name)[0]   # Position du thorax
        self.trajectory.append((fly_pos[0], fly_pos[1]))    # Enregistrer (x, y)

        # ======== Odor detection ========
        if self.count % OLFACTION_RATE == 2:
            olfaction = sim.get_olfaction(sim.fly.name)
            #print(f"olfaction shape: {olfaction.shape}, values: {olfaction}")

            if self.odor_smooth is None:
                self.odor_smooth = olfaction
            else:
                self.odor_smooth = (1 - self.alpha) * self.odor_smooth + self.alpha * olfaction

            self.odor_drives = odor_to_drives(self.odor_smooth) * 3 #fois x to increase the effect of the odor on the speed, otherwise the fly is too much focused on the obstacle avoidance and doesn't move enough towards the target

        # tracking when odor was last sensed 
        mean_odor = np.max(self.odor_smooth) if self.odor_smooth is not None else 0.0
        #print("mean odor is " , mean_odor)
        ODOR_DETECTION_THRESHOLD = 5e-8 
        if mean_odor > ODOR_DETECTION_THRESHOLD: # we smell the source
            self.t_last_odor = self.count
            if self.count < 3 : print("found immediatly")
            self.last_odor_drives = self.odor_drives # will be the our aim if we lose the smell next step

        steps_since_odor = self.count - self.t_last_odor # how long it's been since we lost the smell

        if steps_since_odor <= self.LOST_THRESHOLD: 
            self.odor_state = "TRACKING"
        else: # we lost the smell for too long
            self.odor_state = "LOST"
            if self.wind_state == "INTERFERING": #wind is not forward, so the source is still where we smelled it last
                self.odor_drives = self.last_odor_drives
            else : 
                self.odor_drives = np.array([1.5, 0.5])  # lean right
                self.odor_drives = 0.5*self.odor_drives 
                '''
                if (self.count // 50000) % 2 == 0:
                    self.odor_drives = np.array([1.5, 0.5])  # lean right
                    print('leaning right because I am lost')
                else:
                    self.odor_drives = np.array([0.5, 1.5])  # lean left
                    print('leaning left because I am lost')
                '''
            # even with forward wind we do not smell it, it is really lost, need to search again, we slow down to find it before moving again

        if self.odor_state != self.prev_state: # change of state
            if self.odor_state == "TRACKING":
                print(f"Step {self.count}: Found the smell !") # we had lost it, now we found it
            elif self.odor_state == "LOST":
                print(f"Step {self.count}: Lost the smell :( ")
                #if self.wind_state == "INTERFERING": print('still hope') 
                #else : print('lost for good')
            #print(f"Step {self.count}: {self.odor_state} (mean_odor={mean_odor:.6f})")
            self.prev_state = self.odor_state
            
        # ======== Slope detection ========
        if self.count % PITCH_DETECTION_RATE == 1: # if was 0, the pitch or the derivative pitch would be reset to 0 right before the color_vision() starts
            self.detect_slope_proprioceptive(sim)
            self.compute_convexe_concave()


        # ======== Color vision ========
        if self.count % VISION_RATE == 0:
            im = self.color_vision(sim)
            self.adapts_ROI_to_slope() 

            if self.is_flipped(sim, im):
                self.upside_down = True
                self.recover_fly(im)
                print('mother I fell')
            else :
                self.upside_down = False
    


        # ======== Obstacle / Dragonfly detection ========
            
            self.detect_line_jump(sim) # calls detection of both obstacles and dragonfly
            self.decide_avoidance_strategy()
            self.dragonfly_avoidance_strategy()
            self.show_rectangles(im)


         # ======== Dragonfly avoidance ========
        
        
        # ======== Obstacle avoidance ========
        if self.avoiding_obstacle or self.avoiding_dragonfly:
            if self.avoidance_direction == -1:
                self.turn_left()
            elif self.avoidance_direction == 1:
                self.turn_right()
            
            else :
                self.speed = SUSPICIOUS

        if self.avoiding_dragonfly == True :
                    self.avoiding_obstacle = False


        if self.avoiding_dragonfly and not self.avoiding_obstacle:
            self.speed = RUN_AWAY_FROM_DRAGONFLY

        self.avoidance_timer -= 1
        self.avoidance_dragonfly_timer -= 1

        if self.avoidance_timer <= 0 or self.avoidance_dragonfly_timer <= 0:
                # Fin de l'évitement
                self.avoiding_obstacle = False
                self.avoiding_dragonfly = False
                self.avoidance_direction = 0
                self.no_turn()

        
        # ======== Visualization ========
        flag = True if self.mode == "tuning ROI" or self.mode == "tuning slope" else False

        if self.frames: # because is empty at the first steps
            self.show_ROI_obst(self.frames[-1], show_lines = flag)
            self.show_ROI_drag(self.frames[-1], show_lines = flag)


        # ======== Wind analysis ========
        self.step_count = self.step_count+1
        if self.step_count % 5000 == 0 and self.step_count > 0:
            # get current antenna data
            antenna_data = sim.get_antenna_data(sim.fly.name)
            
            quat_l = antenna_data['l']['qpos']
            quat_r = antenna_data['r']['qpos']
            
            euler_l = Rotation.from_quat([quat_l[1], quat_l[2], quat_l[3], quat_l[0]]).as_euler('xyz', degrees=True)
            euler_r = Rotation.from_quat([quat_r[1], quat_r[2], quat_r[3], quat_r[0]]).as_euler('xyz', degrees=True)
            
            pitch_diff = euler_l[0] - euler_r[0]
            roll_diff  = euler_l[1] - euler_r[1]
            yaw_diff   = euler_l[2] - euler_r[2]
            
            #predicted_direction = wind_analysis(self, pitch_diff, roll_diff, yaw_diff)
            #print(f"Step {self.step_count}: predicted wind direction = {predicted_direction}°, so {self.wind_state}")

        
        # ======== Instructions to body =======
        if self.current_slope_category == "carreful_upsidedown":
            self.speed = 0.2  # almost stop, let physics stabilize
        else:
            self.speed = SUSPICIOUS
        
        if self.upside_down == True :
            # Drive legs asymmetrically to generate a rolling torque
            self.odor_drives = np.array([2.0, 0.0])  # full force one side
            self.speed = SUSPICIOUS * 2
            self.k = TURN_COEFF * 2
            #print('trying my best')

        drives = self.speed * self.odor_drives * np.array([self.k, 1/self.k])
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion
    

###################################################################################
########################### Obstacle Avoidance Strategy ###########################
###################################################################################



    def dragonfly_avoidance_strategy(self):

        """
        Analyse les positions de la libellule et décide si la mouche doit virer à gauche ou à droite pour l'éviter.
        """

        regions = self.where_is_dragonfly()
        
        if np.mean(regions) > 0.25 or np.mean(regions) == 0: # if the dragonfly is detected in multiple regions, we are not sure about where it is and we do not want to take the risk to turn in the wrong direction
            return
        print("dragonfly detected in region(s) :", regions)
        idx_max = np.argmax(regions)
        
        if idx_max == 0 or idx_max == 1:  # Dragonfly à gauche
            self.avoidance_direction = 1
        elif idx_max == 2 or idx_max == 3: # Dragonfly à droite
            self.avoidance_direction = -1

        self.avoiding_dragonfly = True
        self.avoidance_dragonfly_timer = AVOIDANCE_DRAGONFLY_DURATION

 
    def decide_avoidance_strategy(self):
        """
        Analyse les obstacles détectés et décide si la mouche doit virer à gauche ou à droite.
        """
        if self.avoiding_obstacle:
            return
        
        
        regions = np.zeros(NB_OF_RECT)
        regions = self.comparison_obstacles()
        
        max_danger = np.max(regions)
        
        if max_danger > DANGER_THRESHOLD:
            self.obs = True
            idx_max = np.argmax(regions)
            self.danger_zone_index = idx_max
            
            if idx_max == 0 or idx_max == 1:  # Obstacle à gauche
                #print("danger à gauche, je vire à droite", max_danger)

                self.avoidance_direction = 1
                if idx_max == 1: 
                    self.speed = SUSPICIOUS * 0.5 * 1/idx_max # plus l'obstacle est proche du centre, plus la mouche ralentit pour éviter
            else: # Obstacle à droite
                self.avoidance_direction = -1
                #print("danger à droite, je vire à gauche", max_danger)
                if idx_max == 2: 
                    self.speed = SUSPICIOUS * 0.5 *1/idx_max
            
            self.avoiding_obstacle = True
            self.avoidance_timer = AVOIDANCE_DURATION
        
        if len(self.trajectory) > 0:
            current_pos = self.trajectory[-1]
            self.obstacle_detection_points.append(current_pos)
            self.obs = False

    def comparison_obstacles(self):
        """
        This functions analyzes all existing obstacles and decides which one of the 4 rectangular regions is 
        the most dangerous.

        1) The bigger takes-it-all :
        We could check which region posses the heighest obstacle and defining it as the most dangerous region.
        2) The most numerous-er takes-it-all :
        We can add all the heights inside each regions. The region with the highest sum is considered the most dangerous.
        This method is better that the 1) because is an obstacle is close, many ROIs can detect it and it will increase the danger perceive in the corresponding region.
        
        Returns : the np.array containing the sum of heights of each regions.
        """
        regions = np.zeros(NB_OF_RECT)

        for i in range(len(self.last_vertical_segments)): # contains (x, y_top, y_bottom)
            x = self.last_vertical_segments[i][0]
            height = self.last_vertical_segments[i][2]-self.last_vertical_segments[i][1]

            if self.is_left(x):
                if self.is_end_of_roi(x):
                    regions[0] += height
                else:
                    regions[1] += height
            else:
                if self.is_end_of_roi(x):
                    regions[3] += height
                else:
                    regions[2] += height

        idx_max = np.argmax(regions)
        for i in range(NB_OF_RECT):
            if i == idx_max:
                self.all_rectangles[i].color = COLOR_RED
            else:
                self.all_rectangles[i].color = COLOR_BLACK
        return regions

    def turn_right(self): self.k = TURN_COEFF
    def turn_left(self): self.k = 1/TURN_COEFF
    def no_turn(self): self.k = 1            

    def plot_trajectory(self, save_path="trajectory_plot.png"):
        """
        Trace la trajectoire de la mouche en rouge et marque les points 
        de détection d'obstacles en noir.
        """
        import matplotlib.pyplot as plt
        
        if len(self.trajectory) == 0:
            print(" Aucune trajectoire à tracer")
            return
        
        # Extraire x et y de la trajectoire
        x_coords = [pos[0] for pos in self.trajectory]
        y_coords = [pos[1] for pos in self.trajectory]
        
        # Créer la figure
        plt.figure(figsize=(12, 8))
        
        # Tracer la trajectoire en rouge
        plt.plot(x_coords, y_coords, 'r-', linewidth=2, label='Trajectoire de la mouche', alpha=0.7)
        
        # Marquer le point de départ en vert
        plt.plot(x_coords[0], y_coords[0], 'go', markersize=12, label='Départ', zorder=5)
        
        # Marquer le point d'arrivée en bleu
        plt.plot(x_coords[-1], y_coords[-1], 'bs', markersize=12, label='Arrivée', zorder=5)
        
        
        plt.xlabel('Position X (mm)', fontsize=12)
        plt.ylabel('Position Y (mm)', fontsize=12)
        plt.title('Trajectoire de la mouche avec détections d\'obstacles', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Sauvegarder
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Trajectoire sauvegardée : {save_path}")
        print(f"   - Points de trajectoire : {len(self.trajectory)}")
        
        plt.close()
    
        return save_path
    


#########################################################################################
############################## Vision processing functions ##############################
##########################################################################################

    def color_vision(self, sim: MiniprojectSimulation):

        vision_data = sim.get_raw_vision(sim.fly.name)
        im = np.concatenate([vision_data[0], vision_data[1]], axis=1)
        self.frames.append(im)

        return im

    def set_edge_drawing(self, status):
        """Toggle the visibility of detected edges (red points) in the visualization."""
        self.draw_edges = status


##########################################################################################
############################## MODIFYING THE DISPLAY #####################################
##########################################################################################
    def show_rectangles(self, im):
        """ 
        In the middle of the field of view, there is a small region which correspond to where the fly would be if it would still go forward. 
        This region is divided in 4 rectangular regions. 
        The one that has the most dangerous obstacles will appear in red in the vision videoy.
        """
        # Dynamically calculate adaptive vertical bounds based on current ROI positions
        y_top = int(np.clip(self.all_rois_obst[0].y - HEIGHT_INCR, 0, im.shape[0] - 1))
        y_bottom = int(np.clip(self.all_rois_obst[2].y + HEIGHT_INCR, 0, im.shape[0] - 1))

        for rect in self.all_rectangles: # all_rectangles = the 4 "central regions"
            rect.y_top, rect.y_bottom = y_top, y_bottom
            
            if rect.color == COLOR_RED: 
            # Instead of always showing the 4 regions, and only modifying there color, 
            # we decide to show only the regions in red (which contains the most dangerous obstacles.
                im[rect.y_top:rect.y_bottom + 1, rect.x0] = rect.color # Left 
                im[rect.y_top:rect.y_bottom + 1, rect.x1] = rect.color # Right
                im[rect.y_top, rect.x0:rect.x1 + 1] = rect.color       # Top
                im[rect.y_bottom, rect.x0:rect.x1 + 1] = rect.color    # Bottom

        return 0

    def show_ROI_obst(self, im, default_color=COLOR_BLACK, show_lines=False):
        """ 
        The ROIs are horizontal lines in the fly's filed of view. It is along them that the osbatcle detection works.
        
        -Edges :            
            We will detect abrupt varaition of green intensity along these lines ("edges"), to conclude 
            that we may have encountered a boudary between grass and obstacle. These edges are shown in RED.
        -Inner segments :   
            Between 2 edges, there is either the inner part of an obstacle, either something else (ground, 
            sky...). If it is indeed the inner part of an obstacle, it is considered as an "inner segment", and it is shown 
            in GREEN, in the vision video.
        -Heights :
            We approximate the height of an obstacle as the length of a vertical line inside the obstacle, 
            that passes in the center ("starting point") of an inner segment. They are shown in BLUE in the vision video.

        Returns : the last frame ("im") with all of the above elements drawn
        """

        # Draw edges in RED
        for roi in self.all_rois_obst:
            y_draw = int(np.clip(roi.y, 0, im.shape[0]-1))
            if show_lines :
                im[y_draw, roi.x0:roi.x1] = default_color # to see the horizontal black lines along which we detect the edges
            if roi.is_active: # an roi is active is an edge was detected along it
                if self.draw_edges:
                    for i in roi.edge_indices:
                        im[y_draw, (int(roi.x0 + i - roi.diff_width/2)):(int(roi.x0 + i + roi.diff_width/2))] = COLOR_RED
        # Draw inner parts in GREEN
                for start, end in roi.inner_segments:
                    im[y_draw, start:end] = COLOR_GREEN
        
        # Draw the heights in BLUE (drawn last and thicker for visibility)
        for vx, vy0, vy1 in self.last_vertical_segments:
            # vx is the x-pos of the height ; vy0 and vy1 are its vertical boundaries
            vx_start = max(0, vx - 1)
            vx_end = min(im.shape[1], vx + 2)
            im[vy0:vy1, vx_start:vx_end] = COLOR_BLUE
            if self.obs == True:
                im[vy0:vy1, vx_start:vx_end] = COLOR_RED
        return im

    def show_ROI_drag(self, im, default_color=COLOR_BLACK, show_lines=False):
        # Draw dragonfly ROIs in black and edges in GREEN
        img_height, img_width = im.shape[:2]
        
        for roi in self.all_rois_drag:
            y_draw = int(np.clip(roi.y, 0, img_height - 1))
            
            if show_lines:
                # Bounds check for horizontal line
                x0_safe = int(np.clip(roi.x0, 0, img_width))
                x1_safe = int(np.clip(roi.x1, 0, img_width))
                im[y_draw, x0_safe:x1_safe] = default_color
                
            if roi.is_active:
                # Check if edge_indices is not empty to avoid errors
                if not roi.edge_indices:
                    continue
                    
                x_l = int(min(roi.edge_indices))
                x_r = int(1 + max(roi.edge_indices))
                side = int((x_r - x_l) // 2)
                y_t = int(roi.y - side)
                y_b = int(1 + roi.y + side)
                
                # Clamp all boundaries to image dimensions
                # CRITICAL: y_b and x_r are used as indices, so they must be < img_height and img_width
                y_t = int(np.clip(y_t, 0, img_height - 1))
                y_b = int(np.clip(y_b, 0, img_height - 1))
                x_l = int(np.clip(x_l, 0, img_width - 1))
                x_r = int(np.clip(x_r, 0, img_width - 1))
                
                # Ensure y_b and x_r won't cause index errors (they can be at most img_height-1 and img_width-1)
                # since we use them as single indices (im[y_b, ...] and im[..., x_r])
                if y_b >= img_height:
                    y_b = img_height - 1
                if x_r >= img_width:
                    x_r = img_width - 1
                
                # Only draw if we have valid regions
                if y_b > y_t and x_r > x_l:
                    im[y_t:y_b, x_l] = COLOR_BLUE   # Left border
                    im[y_t:y_b, x_r] = COLOR_BLUE   # Right border
                    im[y_t, x_l:x_r] = COLOR_BLUE   # Top border
                    im[y_b, x_l:x_r] = COLOR_BLUE   # Bottom border
                    
                    self.dragonfly_pos = ((y_t + side), (x_l + side))

#######################################################################################
################################# OBSTACLE DETECTION ##################################
#######################################################################################

    def detect_line_jump(self, sim: MiniprojectSimulation):
        """
        Basically this function just calls more "technical" technical functions, for each ROIs.

        Returns : for each ROI if it crosses an obstacle
        """
        im = self.frames[-1] 
        results = []

        # Check which ROI crosses an obstacle. This check also activates other detection functions for each ROIs
        for roi in self.all_rois_obst: 
            detected = self.detect_line_jump_ROI(im, roi) # boolean wether there is an obstacle crossing it or not.
            results.append(detected)

        # Check dragonfly ROIs
        for roi in self.all_rois_drag:
            self.detect_line_jump_ROIdrag(im, roi)

        # Compute the vertical segment (= the approx. height of an obstacle) out of each starting point
        target_points = self.starting_point() # the center of each inner sgements
        self.last_vertical_segments = [] # all the heights
        for vx, vy_start in target_points:
            res = self.detect_vertical_obstacle_bounds(im, vx, vy_start)
            if res is not None:
                self.last_vertical_segments.append(res)
            # A vertical segment is basicall just (x-pos, y-top, y-bottom)

        return tuple(results)

    def detect_line_jump_ROI(self, im, roi):
        """
        Detects multiple edges and evaluates every interval (including those at the start and end of 
        the scanline) for stability and intensity.
        This allows edges to have inner obstacle parts on both left and right.
        """
        y, x0, x1 = int(roi.y), roi.x0, roi.x1
        green_line = im[y, x0:x1, GREEN].astype(int)
        blue_line = im[y, x0:x1, BLUE].astype(int)
        
        roi.edge_indices = []
        roi.inner_segments = []
        
        if len(green_line) < 10:
            return False

        # 1. Find all edge indices using the threshold (threshold = th_line)
        i = 0
        while i < len(green_line) - roi.diff_width:
            if self.different_intensities(green_line[i + roi.diff_width], green_line[i], self.th_line):
                roi.edge_indices.append(i)
                """center = i + diff_width // 2
                # Store this segment
                roi.detected_segments.append((
                    x0 + max(0, center - 5),
                    x0 + min(len(green_line), center + 5)
                ))"""
                i += (roi.diff_width*2) # Skip ahead to find the next discrete edge (not the same one)
            else:
                i += 1

        # 2. Analyze all intervals (start of line, edges, end of line) for 'inner' parts.
        # Each edge boundary allows checking for an inner part to BOTH its left and right.
        boundary_points = [0] + [idx + 2 for idx in roi.edge_indices] + [len(green_line)]
        
        for k in range(len(boundary_points) - 1):
            idx_start = boundary_points[k]
            idx_end = boundary_points[k+1] 
            # k and k+1, because 2 inner segments next to each other is possible, because of the shades 
            # on different faces of an obstacle.
            
            # Avoid checking the pixels immediately inside the edge transition
            check_start = idx_start + (5 if k > 0 else 0)
            check_end = idx_end - (5 if k < len(boundary_points) - 2 else 0)
            
            if check_end > check_start + 2:
                segment = green_line[check_start:check_end]
                blue_segment = blue_line[check_start:check_end]
                # blue segment is just to capture blue values along a segment, for the check in it is in the sky

                # Check for: not in the sky AND not in the grass AND "stable" intensity 
                if not self.in_the_sky(np.mean(blue_segment)) and not self.in_the_grass(np.mean(segment)) and np.ptp(segment) <= 3:
                    roi.inner_segments.append((x0 + idx_start, x0 + idx_end))

        roi.is_active = len(roi.edge_indices) > 0
        return roi.is_active
    

    def detect_vertical_obstacle_bounds(self, im, x, y_start):
        """
        Scans vertically from a seed point to find the upper and lower edges of an obstacle using intensity gradients.
        Keep in mind taht the y-axis is "inverted" so the positive direction is toward the bottom.        

        Returns : the coordinates of the height, corresponding to the starting point passed in the function's arguments.
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
        for y in range(int(y_start), diff, -1): # also use "diff" to go toward a y very high
            
            # Check if we reached either: an edge (this time, we compare to the inital point, so we can detect the end of the obstacle even if the thansition with the grass is smooth) ; the sky ; part that receives no lignt signal
            if self.different_intensities(green_v[y - diff], green_v[int(y_start)], th) or \
               self.in_the_sky(blue_v[y]) or \
               self.in_the_black([red_v[y], green_v[y], blue_v[y]]):
                y_top = y
                break
        
        # Discard if the obstacle only goes downward (it's is to avoid a false positive returnong a very big height as it would just continue in the grass all over the image)
        if y_start - y_top < 13:
            return None
            
        # Downward scan for lower bound
        for y in range(int(y_start), im.shape[0] - diff):

            # Same check. Shouldn't reach sky parts in thsi direction, but we never know, maybe there will be a situation where the blade of grass is very very bent
            if self.different_intensities(green_v[y + diff], green_v[int(y_start)], th) or \
               self.in_the_black([red_v[y], green_v[y], blue_v[y]]):
                y_bottom = y
                break
                
        return (vx, y_top, y_bottom)
    

#########################################################################################################
######################################## DRAGONFLY DETECTION ############################################
#########################################################################################################
    
    
    def detect_line_jump_ROIdrag(self, im, roi):
        """
        Detects dragonfly head boundaries using vectorized color thresholding.
        Efficiently finds regions with high red and low green/blue intensity.
        """
        y, x0, x1 = int(roi.y), roi.x0, roi.x1
        line_rgb = im[y, x0:x1].astype(int)
        
        # Vectorized check: Red > 100, Green < 50, Blue < 50
        is_red = (line_rgb[:, RED] > 100) & (line_rgb[:, GREEN] < 50) & (line_rgb[:, BLUE] < 50)
        
        # Detect boundaries (transitions from False to True or True to False)
        diff = np.diff(is_red.astype(int), prepend=0, append=0)
        roi.edge_indices = np.where(diff != 0)[0].tolist()

        roi.is_active = len(roi.edge_indices) > 0
        return roi.is_active


    def where_is_dragonfly(self):

        regions = np.zeros(NB_OF_RECT)
        x = self.dragonfly_pos[1]
        separation = 900/NB_OF_RECT

        for i in range(NB_OF_RECT):
            if x < (1+i)*separation:
                regions[i] = 1

        if np.mean(regions)>0.25 :
            #print("error : dragonfly is detected in multiple regions")
            return np.zeros(NB_OF_RECT)

        return regions


    print('hihiiii')

#########################################################################################################
####################### FUNCTIONS THAT HELP THE TESTS BETWEEN DIFFERENT INTENSITIES #####################
#########################################################################################################
    def different_intensities(self, int1, int2, epsilon):
        if abs(int1 - int2) > epsilon:
            return True
        return False
    
    def in_the_sky(self, value):
        return value > 10

    def in_the_black(self, point): 
       return all(v < 30 for v in point)

    def capture_mean_grass(self):
        """
        As the fly alwasy look forward, the grass close to it should be similarly exposed to the sun as the grass far away.
        So we use the "close" grass between the feet to capture the mean intensity of grass for further comparison.
        We check really between the fly's feet so that we do not capture anything else than grass (otherwise, the fly has more important 
        problems to take care of).
        """
        im = self.frames[-1]
        mean = 0
        line = np.concatenate([im[400, 300:400, GREEN], im[400, 500:600, GREEN]]).astype(int)
        mean = np.mean(line)

        return mean
    
    def in_the_grass(self, value):
        if self.different_intensities(value, self.capture_mean_grass(), 20):
            return False
        return True


################################################################################################
################################### PROPRIOCEPTIVE FUNCTIONS ###################################
################################################################################################

    def detect_slope_proprioceptive(self, sim: MiniprojectSimulation):
        from scipy.spatial.transform import Rotation
        
        # Get thorax infos (1st body = index 0)
        body_rotations = sim.get_body_rotations(sim.fly.name)
        # Reorder MuJoCo [w, x, y, z] to SciPy [x, y, z, w]
        thorax_quat = body_rotations[0][[1, 2, 3, 0]]
        
        # Converts quaternion -> Euler angles
        rotation = Rotation.from_quat(thorax_quat)
        euler_angles = rotation.as_euler('xyz')  # [roll, pitch, yaw]
        pitch = euler_angles[1]  # pitch ~[-0.5, 0.5]
        
        # Some subjective qualifications
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
    
    def compute_convexe_concave(self):
        window = (VISION_RATE/PITCH_DETECTION_RATE+1)
        if self.pitch_count < window:
            self.pitch_derivative += ((self.prev_pitch - self.pitch)/self.speed)
            self.pitch_count += 1
        if self.pitch_count == window : self.pitch_count = 1
        return 0
    
    def is_flipped(self, sim: MiniprojectSimulation, im):
        """
        Returns : just a boolean wether the fly is flipped or not
        """
        for roi in self.all_rois_obst:
            blue_line = im[int(roi.y), roi.x0:roi.x1, BLUE].astype(int)
            if np.any(blue_line > 0):
                if self.maybe_flipped >= FLIPPED_FOR_SURE:   
                    self.maybe_flipped = 0
                    return True
                
                else:   
                    self.maybe_flipped += 1
        return False
    
    def recover_fly(self, im):
        im[110:160, 110:160] = COLOR_BLACK # just ldisplays a small dark square 
        return 0
    
    def adapts_ROI_to_slope(self): 
        """
        If the ROIs were static in front of the fly's eyes, we would often look at grass or sky. So we adapt
        the vertical position of the ROIs.
        When the fly is ascending -> ROI needs to be little higher (and vice versa).

        As described in the report, there is 2 ways to adapt the ROI, both related to the path of the fly. 
        1) We look at the derivative of the path = the slope the fly feels = the pitch
        2) We look wether the fly is currently walking on a concave or a convex path = derivative of the pitch

        The second option can be activated when using the mode below in the run_controller. But the approximation 
        of the derivative of the pitch was too not accurate enough so we kept the first option by default. 
        """

        coeff = self.pitch_derivative if self.mode == "adapts_ROI derivative" else self.pitch
        coeff *= self.pitch_weight

        for roi in self.all_rois_obst:
            new_y = roi.base_y + coeff
            roi.y = np.clip(new_y, 75, 475)
            
        for roi in self.all_rois_drag:
            new_y = roi.base_y + coeff*1.5
            roi.y = np.clip(new_y, 1, 499)
            
        self.pitch_derivative = 0
        return 0
    

#############################################################################################################  
####################################### POSITIONNAL TESTS ###################################################
#############################################################################################################
    def starting_point(self):
        """
        Identifies the x-coordinates of the midpoints of the 'inner obstacle' segments
        
        Returns:
            list: List of (x, y) coordinates of the seed points.
        """
        center_vision_x = (MIDDLE_LEFT + MIDDLE_RIGHT) // 2
        points = []

        for roi in self.all_rois_obst:
            best_left = None
            best_right = None
            min_dist_left = float('inf')
            min_dist_right = float('inf')

            for start, end in roi.inner_segments: # an inner segment is just 2 x-coordinates
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
        """
        As said in show_ROI_obst, for obstacles detection, only the center part of the fly's field of view 
        is interesiting. This center part is itself divided in 4 rectangular regions. As we have 4 regions, 
        it is better in terms of computational power to have only 2 boolean returning functions to determine 
        in which region belongs a certain point.
        """
        return x <= (MIDDLE_LEFT)
    
    def is_end_of_roi(self, x):
        """
        Same as description of is_left().
        """
        return x <= ((MIDDLE_LEFT + X_LEFT)/2) or x >= ((MIDDLE_RIGHT + X_RIGHT)/2)
    
    def is_in_dragonfly(self, rgb):
        r = rgb[0]
        g = rgb[1]
        b = rgb[2]

        if r > 100 and g < 50 and b < 50:
            return True
        return False

##########################################################################################
############################## Odor processing functions #################################
##########################################################################################

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



##########################################################################################
############################### Wind analysis functions ##################################
##########################################################################################
def wind_analysis(self, pitch_diff, roll_diff, yaw_diff) :
    wind_direction = 0
    if pitch_diff < -15 : 
        wind_direction = 0
        self.wind_state = "INTERFERING" # the wind interfers with the smell if we are moving towards the source
    elif pitch_diff > -8 : 
        wind_direction = 180
        self.wind_state = "SILENT" # frontward wind, we can still smell the source if we are moving towards it
    else :
        if roll_diff > 0:
            if roll_diff < 1 : 
                wind_direction = 270
                self.wind_state = "INTERFERING"
            else :
                if yaw_diff > 0 : 
                    wind_direction = 45
                    self.wind_state = "INTERFERING"
                else : 
                    wind_direction = 135
                    self.wind_state = "SILENT"
        if roll_diff < 0:
            if roll_diff > -1 : 
                wind_direction = 90
                self.wind_state = "INTERFERING"
            else :
                if yaw_diff > 0 : 
                    wind_direction = 315
                    self.wind_state = "INTERFERING"
                else : 
                    wind_direction = 225
                    self.wind_state = "SILENT"
    return wind_direction
