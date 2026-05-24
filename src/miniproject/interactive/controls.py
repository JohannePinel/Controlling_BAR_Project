from abc import ABC, abstractmethod

import pygame


class Controls(ABC):
    @abstractmethod
    def __init__(self, game_state):
        """Initialize controls"""
        pass

    @abstractmethod
    def process_events(self, events):
        """Consume input events for one frame"""
        pass

    @abstractmethod
    def any_key_pressed(self):
        """Return whether any control key is currently pressed"""
        pass

    @abstractmethod
    def get_actions(self):
        """Translate keys into simulator actions"""
        pass

    def quit(self):
        """Quit and cleans control handles threads ect"""
        pass

    def flush_keys(self):
        """Flush the keys"""
        pass


class KeyboardControl:
    def __init__(self, game_state, control_mode="sticky"):
        self.game_state = game_state
        self.is_joystick = False
        self.control_mode = control_mode
        self.hold_to_move = control_mode == "hold"

        self.key_forward = pygame.K_w
        self.key_backward = pygame.K_s
        self.key_left = pygame.K_a
        self.key_right = pygame.K_d
        self.key_stop = pygame.K_q
        self.key_acceleration = pygame.K_f
        self.key_odor_mode = pygame.K_SPACE
        self.key_change_banana_pos = pygame.K_c

        self.prev_gain_left = 0.0
        self.prev_gain_right = 0.0

        self.prev_odor_mode = False
        self.toggle_odor = False

        self.prev_acceleration = 1.0
        self.toggle_acceleration = False

        self.banana_pos_just_changed = False
        self.change_banana_pos = False
        

    def process_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                self.game_state.set_quit(True)
            elif event.type == pygame.KEYDOWN:

                if event.key == self.key_odor_mode: 
                    self.toggle_odor = True 
                if event.key == self.key_change_banana_pos:
                    self.change_banana_pos = True
                if event.key == self.key_acceleration:
                    self.toggle_acceleration = True

                if event.key == pygame.K_ESCAPE:
                    self.game_state.set_quit(True)
                elif event.key == pygame.K_r and not self.game_state.get_reset():
                    self.game_state.set_reset(True)
                

    def any_key_pressed(self):
        keys_pressed = pygame.key.get_pressed()
        # pygame.key.get_pressed() is just a list of bool for each key on the laptop
        # So keys_pressed[self.key_forward] is juste checking this big list at the index "self.key_forward"
        return (
            keys_pressed[self.key_forward]
            or keys_pressed[self.key_backward]
            or keys_pressed[self.key_left]
            or keys_pressed[self.key_right]
            or keys_pressed[self.key_stop]
            #or keys_pressed[self.key_acceleration]
            #or keys_pressed[self.key_odor_mode]
        )

    def get_actions(self, keys_pressed=None, memory=False):
        gain_left = self.prev_gain_left if memory else 0.0
        gain_right = self.prev_gain_right if memory else 0.0

        new_odor_mode = self.prev_odor_mode
        new_acceleration = self.prev_acceleration
        order_changing_banana_pos = self.banana_pos_just_changed

        command_applied = False
        acceleration_applied = False


        # SECURITY
        if keys_pressed is None:
            keys_pressed = pygame.key.get_pressed()
        if keys_pressed[self.key_stop]:
            gain_left = 0.0
            gain_right = 0.0
            command_applied = True

        # ALL TOGGLES : 1st phase
        elif self.toggle_odor:
            new_odor_mode = not self.prev_odor_mode # 1st we toggle
            self.toggle_odor = False # Can't re-toggle before key is released
        elif self.toggle_acceleration:
            new_acceleration = 1.5 if self.prev_acceleration == 1.0 else 1.0 
            self.toggle_acceleration = False 

        # ALL TOGGLES : 2nd phase
        if new_odor_mode == True: 
            gain_right = 1.0
            gain_left  = 1.0
            command_applied = True

        elif keys_pressed[self.key_left] and not keys_pressed[self.key_right]: # means that we switch OUT of odor mode
            if self.prev_gain_left <= 0 or self.prev_gain_right <= 0:
                gain_left = -1.2
                gain_right = 1.5
            else:
                gain_left = 0.4
                gain_right = 1.2
            command_applied = True
        elif keys_pressed[self.key_right] and not keys_pressed[self.key_left]:
            if self.prev_gain_left <= 0 or self.prev_gain_right <= 0:
                gain_left = 1.5
                gain_right = -1.2
            else:
                gain_left = 1.2
                gain_right = 0.4
            command_applied = True
        elif keys_pressed[self.key_forward] and not keys_pressed[self.key_backward]:
            gain_right = 1.0
            gain_left = 1.0
            command_applied = True
        elif keys_pressed[self.key_backward] and not keys_pressed[self.key_forward]:
            gain_right = -1.0
            gain_left = -1.0
            command_applied = True

        """ I comment it just for when I train the fly to always follow the odor so by default it needs gains of 1.0
            
            if self.hold_to_move and not command_applied and not acceleration_applied:
            gain_left = 0.0
            gain_right = 0.0"""
        
        if self.change_banana_pos:
            order_changing_banana_pos = not self.banana_pos_just_changed # 1st we toggle
            self.change_banana_pos = False # Can't re-toggle before key is released

        if new_odor_mode != self.prev_odor_mode:
            print(f"new_odor_mode: {new_odor_mode}")
        self.prev_odor_mode = new_odor_mode

        if new_acceleration != self.prev_acceleration:
            print(f"in controls.py, gain_left*acc: {gain_left*new_acceleration} -- gain_right*acc: {gain_right*new_acceleration}")
        self.prev_acceleration = new_acceleration


        self.prev_gain_left = gain_left*new_acceleration
        self.prev_gain_right = gain_right*new_acceleration
        return gain_left*new_acceleration, gain_right*new_acceleration, new_odor_mode, order_changing_banana_pos

    def flush_keys(self):
        return

    def quit(self):
        self.game_state.set_quit(True)
