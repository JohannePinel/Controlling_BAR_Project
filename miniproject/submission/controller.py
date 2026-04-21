import numpy as np
from miniproject.simulation import MiniprojectSimulation


class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        # you may also implement your own turning controller
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)
        
        self.frames = [] # must be out of step()
        self.count = 0
        self.height_obstacles = 400 # manually tuned
        self.starts_scanning = 200 # manually tuned
        self.starts_comparing = 360 # manually tuned
        self.different_objects = 20 # manually tuned  
        self.color_of_grass = 90 # manually tuned

    def step(self, sim: MiniprojectSimulation):
        self.count += 1
        # implement your control algorithm here
        olfaction = sim.get_olfaction(sim.fly.name)

        # color vision
        if self.count % 100 == 0:
            self.im = np.concatenate(
                [
                    sim.get_raw_vision(sim.fly.name)[0],
                    sim.get_raw_vision(sim.fly.name)[1]
                    
                ], axis=1
            )
            self.frames.append(self.im)

        # get other observations as needed
        drives = np.array([0.2, 0.2])  # replace with your control logic
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion

###########################################################
#################### TEST ADRI ############################
###########################################################
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
    
    def checks_for_ostacles(self, sim: MiniprojectSimulation, ):
        intensities = [] # length = nb of pixels checked ; (r, g, b) for each pixel
        obstacle_pos = 0 # on the hizontal axis at self.height_obstacles 


        for i in range(900-(2*self.starts_scanning)): # frames are 900 wide
            pixel = self.frames[-1][self.height_obstacles,i]

            if self.worth_analysing(sim, pixel, i)[1]: # make sure we are not in the black spot
                intensities.append(self.frames[-1][self.height_obstacles, self.starts_scanning+i])
            
            if self.worth_analysing(sim, pixel, i)[0]: # enough datas to have a good mean to compare the pixel to
                if np.abs(intensities[-1] - np.mean(intensities[:-1])) > self.different_objects : 
                    obstacle_pos = (self.starts_scanning + i)
                    break


        return obstacle_pos
    

    def worth_analysing(self, sim: MiniprojectSimulation, pixel, horizontal_pos):
        result = [False, False]
        
        if (horizontal_pos + self.starts_scanning) > self.starts_comparing :
            result[0] = True # already enough data to have a good mean to compare the pixel to

        if pixel[1]>self.color_of_grass:  
            result[1] = True # not on the black spot
         
        return result
