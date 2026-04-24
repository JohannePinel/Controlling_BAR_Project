import numpy as np
from miniproject.simulation import MiniprojectSimulation

GREEN = 1
TURN_RIGHT = 1.25
TURN_LEFT = 1/TURN_RIGHT

class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        # you may also implement your own turning controller
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)
        
        # var pr la vision
        self.frames = []

        # param obstacles
        self.count = 0
        self.last_mean_left = 128
        self.last_mean_right = 128
        self.th_right = 12
        self.th_left = 12
        self.vertl = 330
        self.vertr = 620
        
        # param ROI
        self.a1 = -0.33
        self.a2 = 0.33
        self.b1 = 485
        self.b2 = 185

        # inhibitateur de marche
        self.k = 1


    def step(self, sim: MiniprojectSimulation):
        self.count += 1
        # implement your control algorithm here
        #olfaction = sim.get_olfaction(sim.fly.name)

        # color vision
        if self.count % 100 == 0:
            #self.color_vision(sim) 
            self.ommatidia_vision(sim)
            
            #if self.count % 400 == 0:
                #self.show_ROI(sim) 
            #self.show_ROI(sim)
            #self.checks_for_ostacles(sim)
            #self.checks_height(sim)


        # get other observations as needed
        drives = np.array([1.0*self.k, 1.0/self.k])  # replace with your control logic
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion

###########################################################
#################### TEST ADRI ############################
###########################################################
    
    def color_vision(self, sim: MiniprojectSimulation):
        vision_data = sim.get_raw_vision(sim.fly.name)
        im = np.concatenate([vision_data[0], vision_data[1]], axis=1)

        new_mean_left, new_mean_right = self.mean_green_inside_ROI(im) 

        if np.abs(new_mean_left - self.last_mean_left) > self.th_left: 
            """if self.k>0:
                print("obstacle at left")
                self.k = 0 # stop walking if obstacle detected"""
            self.last_mean_left = new_mean_left
        
        if np.abs(new_mean_right - self.last_mean_right) > self.th_right:
            """if self.k>0:
                print("obstacle at right")
                self.k = 0 # stop walking if obstacle detected"""
            self.last_mean_right = new_mean_right

        self.frames.append(im)
        return 0
    
    def ommatidia_vision_p(self, sim: MiniprojectSimulation):
        vision_data = sim.get_ommatidia_readouts(sim.fly.name)
        im = np.concatenate([self.retina.hex_pxls_to_human_readable(eye.max(-1), color_8bit=True)
                for eye in vision_data], axis=1)

        self.frames.append(im)
        return 0

    def ommatidia_vision(self, sim: MiniprojectSimulation):
        vision_data = sim.get_ommatidia_readouts(sim.fly.name)
        retina = sim.world.fly_lookup[sim.fly.name].retina
        im = np.concatenate(
            [retina.hex_pxls_to_human_readable(eye.max(-1), color_8bit=True)
            for eye in vision_data],
            axis=1,
        )
        self.frames.append(im)
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
        return (self.a1*x1 + self.b1)
    def f2(self, x1):
        return (self.a2*x1 + self.b2)

    def show_ROI(self, sim: MiniprojectSimulation):
        """
        for x in range(900):
            for y in range(512):
                if np.abs(y - self.f1(x))<(0.33) or np.abs(y - self.f2(x))<(0.33) or x==290 or x==620:
                    self.frames[-1][y, x] = [255, 0, 0]
        """

        frame = self.frames[-1]
        height, width = frame.shape[:2]
        
        # Créer des grilles de coordonnées
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Créer le masque ROI (les points qui satisfont les conditions)
        roi_mask = (
            (np.abs(y_coords - self.f1(x_coords)) < 0.33) |  # ligne 1
            (np.abs(y_coords - self.f2(x_coords)) < 0.33) |  # ligne 2
            (x_coords == self.vertl) |                               # colonne 1
            (x_coords == self.vertr)                                 # colonne 2
        )
        
        # Appliquer la couleur rouge à tous les pixels du masque en une seule opération
        frame[roi_mask] = [255, 0, 0]


    
    def is_inside_ROI(self, x, y):
        if np.abs(y - self.f1(x))<(0.33) and np.abs(y - self.f2(x))<(0.33) and x==self.vertl and x==self.vertr:
            return True
        else:
            return False
    
    def mean_green_inside_ROI(self, im):
        green_channel = im[:, :, GREEN]
        
        # ROI gauche [self.vertl:466, 350:451]
        roi_left = green_channel[350:451, self.vertl:466]
        mean_left = np.mean(roi_left[roi_left != 0]) if np.any(roi_left != 0) else 128
        
        # ROI droite [self.vertr:621, 350:451]
        roi_right = green_channel[350:451, self.vertr:621]
        mean_right = np.mean(roi_right[roi_right != 0]) if np.any(roi_right != 0) else 128
        
        return mean_left, mean_right


                
