import numpy as np
from miniproject.simulation import MiniprojectSimulation

GREEN = 1

COLOR_RED = [255, 0, 0]
COLOR_GREEN = [0, 255, 0]
COLOR_BLUE = [0, 0, 255]
COLOR_BLACK = [0, 0, 0]

TURN_RIGHT = 1.25
TURN_LEFT = 1/TURN_RIGHT

class Controller:
    def __init__(self, sim: MiniprojectSimulation, threshold):
        # you may also implement your own turning controller
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)
        
        # var pr la vision
        self.frames = []

        # param obstacles
        self.count = 0
        self.last_mean_left = 128
        self.last_mean_right = 128
        self.th_right = threshold
        self.th_left = threshold
        
        # param ROI
        self.a1 = -0.22
        self.a2 = 0.22
        self.b1 = 440
        self.b2 = 240
        self.vertl = 290
        self.vertm = 450
        self.vertr = 620
        self.widthROI = 40
        self.colorROI = COLOR_BLUE

        # inhibitateur de marche
        self.k = 1

        # param ommatidia 
        self.limR = 75
        self.limH = 125
        self.limM = 250
        self.limB = 375

        # var ommatidia
        self.intensityH = 0
        self.intensityB = 0
        self.ratio = [0, 0, 0]
        self.last_ratios = 6 



    def step(self, sim: MiniprojectSimulation):
        self.count += 1
        #olfaction = sim.get_olfaction(sim.fly.name)

        # color vision
        if self.count % 200 == 0:
            self.color_vision(sim)
            self.detect_mean_variation(sim)
            #self.show_ROI(sim, self.frames[-1], left=255, right=255, fullfill=True)
            #self.ommatidia_vision(sim)
            #visualize_data_ommatidia(sim)


        drives = np.array([1.0*self.k, 1.0/self.k])  
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

    def show_ROI(self, sim: MiniprojectSimulation, im, left, right, fullfill=False):
        
        # left eye
        if left:
            for x in range(self.vertl, self.vertm):
                for y_inc in range(self.widthROI):
                    if fullfill or (y_inc == 0 or y_inc == self.widthROI-1):
                        im[int(round(self.f1(x-y_inc))), x] = [min(left, 255), 0, 0]
        # right eye
        if right:
            for x in range(self.vertm, self.vertr):
                for y_inc in range(self.widthROI):
                    if fullfill or (y_inc == 0 or y_inc == self.widthROI-1):
                        im[int(round(self.f2(x+y_inc))), x] = [min(right, 255), 0, 0]
        return im
    
    def f1(self, x1):
        return (self.a1*x1 + self.b1)
    def f2(self, x1):
        return (self.a2*x1 + self.b2)
    
    def detect_mean_variation(self, sim: MiniprojectSimulation):
        im = self.frames[-1]
        new_mean_left, new_mean_right = self.mean_green_inside_ROI(im)

        if np.abs(new_mean_left - self.last_mean_left) > self.th_left:
            self.obstacle_at_left(sim, im, new_mean_left)
            self.last_mean_left = new_mean_left

        if np.abs(new_mean_right - self.last_mean_right) > self.th_right:
            self.obstacle_at_right(sim, im, new_mean_right)
            self.last_mean_right = new_mean_right

        return new_mean_left, new_mean_right
    
    def mean_green_inside_ROI(self, im):
        """
        Calcule la moyenne de l'intensité du canal vert pour chaque ROI
        (oeil gauche et oeil droit)
        
        Returns:
            Tuple[float, float]: (moyenne_gauche, moyenne_droite)
        """
        mean_green_ROI_left = 0
        mean_green_ROI_right = 0
        count_left = 0
        count_right = 0
        
        # Left eye
        for x in range(self.vertl, self.vertm):
            for y_inc in range(self.widthROI):
                y = int(round(self.f1(x - y_inc)))
                if 0 <= y < im.shape[0]:   
                    count_left += 1
                    # mean_green_ROI_left += im[y, x, GREEN]  
                    mean_green_ROI_left += (im[y, x, GREEN]-mean_green_ROI_left)/count_left # directly constitute the mean

        # Right eye
        for x in range(self.vertm, self.vertr):
            for y_inc in range(self.widthROI):
                y = int(round(self.f2(x + y_inc)))
                if 0 <= y < im.shape[0]:    
                    count_right += 1
                    # mean_green_ROI_right += im[y, x, GREEN]  
                    mean_green_ROI_right += (im[y, x, GREEN]-mean_green_ROI_right)/count_right # directly constitute the mean
        
        # only if we don't use the trick 
        """
        if count_left > 0:
            mean_green_ROI_left = mean_green_ROI_left / count_left
        if count_right > 0:
            mean_green_ROI_right = mean_green_ROI_right / count_right
        """
        return mean_green_ROI_left, mean_green_ROI_right
    
    def obstacle_at_left(self, sim: MiniprojectSimulation, im, new_mean_left):
        self.show_ROI(sim, im, left=new_mean_left, right=False, fullfill=True)
        return 0
    
    def obstacle_at_right(self, sim: MiniprojectSimulation, im, new_mean_right):
        self.show_ROI(sim, im, left=False, right=new_mean_right, fullfill=True)
        return 0
    
    """
    can_walk = False

    def stops_analysing(self, sim: MiniprojectSimulation):
        self.can_walk = True
        return 0
    
    def starts_analysing(self, sim: MiniprojectSimulation):
        self.can_walk = False
        return 0
    
    def looks_for_bananas(self, sim: MiniprojectSimulation):
        self.can_walk = False

        return 0
    
    def detects_horizon_while_stopped(self, sim: MiniprojectSimulation, screenshot):
        self.can_walk = False

        height_of_horizon = 0
        for i in range(5):
            if screenshot[261+i, 100][1] == 128:
                height_of_horizon = 261+i
                break
        self.can_walk = True
        return height_of_horizon

    def checks_for_ostacles(self):
        self.intensities = [] # length = nb of pixels checked ; (r, g, b) for each pixel
        self.obstacle_pos = 0 # on the hizontal axis at self.height_obstacles 

        sum_green_intensities = 128

        for i in range(self.frame_width - (2*self.starts_scanning)): # frames are 900 wide
            pixel = self.frames[-1][self.height_obstacles, i]

            if self.worth_analysing(sim, pixel, i)[1]: # make sure we are not in the black spot
                self.intensities.append(self.frames[-1][self.height_obstacles, self.starts_scanning+i])

            if pixel[GREEN] >= (sum_green_intensities/(i+1)): 
                self.frames[-1][self.height_obstacles, i] = [128, 0, 0] 
            else:
                sum_green_intensities += pixel[GREEN]
            
            if self.worth_analysing(sim, pixel, i)[0]: # enough datas to have a good mean to compare the pixel to
                if np.abs(self.intensities[-1][GREEN] - np.mean(self.intensities[:-1][GREEN] )) > self.different_objects: 
                    self.obstacle_pos = (self.starts_scanning + i)
                    self.frames[-1][self.height_obstacles, self.starts_scanning + i] = [255, 0, 0] # for debugg

                    break 
            
        print("self.intensities[-1]", self.intensities[-1])
        print("self.intensities[-1][GREEN]", self.intensities[-1][GREEN])
        print("diff with mean", np.abs(self.intensities[-1][GREEN] - np.mean(self.intensities[:-1][GREEN])) )
        
        return 0
        """
    
    def checks_height(self, sim: MiniprojectSimulation):

        for i in range(900): # frames are 900 wide
            if np.any(self.frames[-1][self.height_obstacles, i] != [255, 0, 0]):
                self.frames[-1][self.height_obstacles, i] = [0, 0, 128] 
        return 0
    

    def ommatidia_vision(self, sim: MiniprojectSimulation):
        vision_data = sim.get_ommatidia_readouts(sim.fly.name)
        retina = sim.world.fly_lookup[sim.fly.name].retina
        im = np.concatenate(
            [retina.hex_pxls_to_human_readable(eye.max(-1), color_8bit=True)
            for eye in vision_data],
            axis=1,
        )
        self.tilt(sim, im)
        self.frames.append(self.apply_grid(sim, im))
        return 0
    
    def visualize_data_ommatidia(self, sim: MiniprojectSimulation):
        vision_data = sim.get_ommatidia_readouts(sim.fly.name)
        retina = sim.world.fly_lookup[sim.fly.name].retina
        im = np.concatenate(
            [retina.hex_pxls_to_human_readable(eye.max(-1), color_8bit=True)
            for eye in vision_data],
            axis=1,
        )
        im[:, 675:900] = 0
        im[275, :] = 0
        im[300, 25:] = 0
        im[325, 50:] = 0
        im[350, 75:] = 0
        im[375, 100:] = 0
        self.frames.append(im)
        return 0
    
    def apply_grid(self, sim: MiniprojectSimulation, im):
        im[self.limM, :] = 0
        im[self.limH, 0:self.limR] = 0
        im[self.limB, 0:self.limR] = 0

        im[100:400, self.limR] = 0

        return im
    
    def tilt(self, sim: MiniprojectSimulation, im):
        for x in range(self.limR):
            for y in (self.limH , self.limM):
                self.intensityH += 0.01*im[y, x]
            
            for y in (self.limM , self.limB):
                self.intensityB += 0.01*im[y, x]
        
        self.ratio.append([self.intensityH, self.intensityB, self.intensityH/(self.intensityB+1e-5)])
        self.intensityH = 0
        self.intensityB = 0

        return 0

    def last_ratios_mean(self):
        
        return np.mean([x[2] for x in self.ratio[-self.last_ratios:]])
