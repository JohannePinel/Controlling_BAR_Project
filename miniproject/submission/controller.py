import numpy as np
from miniproject.simulation import MiniprojectSimulation

GREEN = 1
a1 = -0.33
a2 = 0.33
b1 = 497
b2 = 199

class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        # you may also implement your own turning controller
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)
        
        # color vision
        self.frames = [] # must be out of step()
        self.count = 0

        # obstacles detection
        self.height_obstacles = 350 # manually tuned
        self.starts_scanning = 200 # manually tuned
        self.starts_comparing = 360 # manually tuned
        self.different_objects = 6 # manually tuned  
        self.color_of_grass = 90 # manually tuned
        self.frame_width = 900 # frames are 900 wide

        self.intensities = [] # length = nb of pixels checked ; (r, g, b) for each pixel
        self.obstacle_pos = 0 # on the hizontal axis at self.height_obstacles 
        self.last_mean_left = 128
        self.last_mean_right = 128


    def step(self, sim: MiniprojectSimulation):
        self.count += 1
        # implement your control algorithm here
        olfaction = sim.get_olfaction(sim.fly.name)

        # color vision
        if self.count % 100 == 0:
            self.color_vision(sim) 
            self.show_ROI(sim)
            #self.checks_for_ostacles(sim)
            #self.checks_height(sim)


        # get other observations as needed
        drives = np.array([1.0, 1.0])  # replace with your control logic
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion

###########################################################
#################### TEST ADRI ############################
###########################################################
    
    def color_vision(self, sim: MiniprojectSimulation):
        self.im = np.concatenate(
            [
                sim.get_raw_vision(sim.fly.name)[0],
                sim.get_raw_vision(sim.fly.name)[1]
                
            ], axis=1
        )
        new_mean_left, new_mean_right = self.mean_green_inside_ROI(self.im) 

        if np.abs(new_mean_left - self.last_mean_left) > 6: # 6 manually tuned
            print("obstacle at left")
            self.last_mean_left = new_mean_left
        
        if np.abs(new_mean_right - self.last_mean_right) > 6:
            print("obstacle at right")
            self.last_mean_right = new_mean_right

        self.frames.append(self.im)
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

    def f1(self, x1):
        return (a1*x1 + b1)
    def f2(self, x1):
        return (a2*x1 + b2)

    def show_ROI(self, sim: MiniprojectSimulation):
        for x in range(900):
            for y in range(512):
                if np.abs(y - self.f1(x))<(0.33) or np.abs(y - self.f2(x))<(0.33) or x==290 or x==620:
                    self.frames[-1][y, x] = [255, 0, 0]
    
    def is_inside_ROI(self, x, y):
        if np.abs(y - self.f1(x))<(0.33) and np.abs(y - self.f2(x))<(0.33) and x==290 and x==620:
            return True
        else:
            return False
    
    def mean_green_inside_ROI(self, im):
        intensities_green_inside_ROI_left = 0
        intensities_green_inside_ROI_right = 0
        div = 1

        for x in range(290, 466): 
            for y in range(350, 451):
                if self.is_inside_ROI(x, y) and im[y, x][GREEN] != 0: 
                    intensities_green_inside_ROI_left += (im[y, x][GREEN])
                    div += 1

        for x in range(466, 621): 
            for y in range(350, 451):
                if self.is_inside_ROI(x, y) and im[y, x][GREEN] != 0: 
                    intensities_green_inside_ROI_right += (im[y, x][GREEN])

        return intensities_green_inside_ROI_left/div, intensities_green_inside_ROI_right/div


                
