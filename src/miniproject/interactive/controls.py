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
        self.accelerate = pygame.K_f

        self.prev_gain_left = 0.0
        self.prev_gain_right = 0.0
        self.acceleration = 1.0
        

    def process_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                self.game_state.set_quit(True)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.game_state.set_quit(True)
                elif event.key == pygame.K_SPACE and not self.game_state.get_reset():
                    self.game_state.set_reset(True)

    def any_key_pressed(self):
        keys_pressed = pygame.key.get_pressed()
        return (
            keys_pressed[self.key_forward]
            or keys_pressed[self.key_backward]
            or keys_pressed[self.key_left]
            or keys_pressed[self.key_right]
            or keys_pressed[self.key_stop]
        )

    def get_actions(self, keys_pressed=None, memory=False):
        gain_left = self.prev_gain_left if memory else 0.0
        gain_right = self.prev_gain_right if memory else 0.0
        acceleration = self.acceleration
        command_applied = False
        acceleration_applied = False

        if keys_pressed is None:
            keys_pressed = pygame.key.get_pressed()

        if keys_pressed[self.accelerate]:
            acceleration += 1.5
            if acceleration > 2.5:
                acceleration = 1.0
        acceleration_applied = True

        if keys_pressed[self.key_stop]:
            gain_left = 0.0
            gain_right = 0.0
            command_applied = True
        elif keys_pressed[self.key_left] and not keys_pressed[self.key_right]:
            if self.prev_gain_left < 0 or self.prev_gain_right < 0:
                gain_right = -0.6
                gain_left = -1.2
            else:
                gain_left = 0.4
                gain_right = 1.2
            command_applied = True
        elif keys_pressed[self.key_right] and not keys_pressed[self.key_left]:
            if self.prev_gain_left < 0 or self.prev_gain_right < 0:
                gain_left = -0.6
                gain_right = -1.2
            else:
                gain_right = 0.4
                gain_left = 1.2
            command_applied = True
        elif keys_pressed[self.key_forward] and not keys_pressed[self.key_backward]:
            gain_right = 1.0
            gain_left = 1.0
            command_applied = True
        elif keys_pressed[self.key_backward] and not keys_pressed[self.key_forward]:
            gain_right = -1.0
            gain_left = -1.0
            command_applied = True

        if self.hold_to_move and not command_applied and not acceleration_applied:
            gain_left = 0.0
            gain_right = 0.0

        print(f"acceleration: {acceleration}")
        self.prev_gain_left = gain_left*acceleration
        self.prev_gain_right = gain_right*acceleration
        self.acceleration = acceleration

        return gain_left*acceleration, gain_right*acceleration

    def flush_keys(self):
        return

    def quit(self):
        self.game_state.set_quit(True)
