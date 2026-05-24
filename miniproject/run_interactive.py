import argparse

import numpy as np
import pygame

from flygym.compose import ActuatorType
from flygym.examples.locomotion import TurningController
from miniproject.interactive import GameState
from miniproject import MiniprojectSimulation

WINDOW_NAME = "COBAR 2026 Miniproject"

# Odor detection parameters
OLFACTION_RATE = 10
DISCARD_VALUE = 123456789
ALPHA = 0.1   
ODOR_DETECTION_THRESHOLD = 1e-8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--keyboard-mode",
        choices=["hold", "sticky"],
        default="hold",
        help=(
            "Keyboard control mode: 'sticky' keeps last gait command until changed, "
            "'hold' only walks while movement keys are actively pressed."
        ),
    )
    parser.add_argument(
        "-l",
        "--level",
        type=int,
        default=4,
        help="The level of the simulation to run. Default is 0.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="The random seed for the simulation. Default is 0.",
    )
    parser.add_argument(
        "--dont-use-pygame-rendering",
        action=argparse.BooleanOptionalAction,
        help=(
            "If experiencing rendering issues, set this option to use opencv rendering instead of pygame."
            "Also requires installing the pynput library."
        ),
    )
    parser.add_argument(
        "--render-fly-vision",
        action=argparse.BooleanOptionalAction,
        help="Whether to also render what the fly sees from its perspective.",
    )
    parser.add_argument(
        "--hills",
        action=argparse.BooleanOptionalAction,
        help=(
            "enable hills"
        ),
    )
    parser.add_argument(
        "--grass",
        action=argparse.BooleanOptionalAction,
        help="enable obstacles",
    )
    parser.add_argument(
        "--wind",
        action=argparse.BooleanOptionalAction,
        help="enable wind",
    )
    parser.add_argument(
        "--dragonfly",
        action=argparse.BooleanOptionalAction,
        help="enable dragonfly",
    )
    return parser.parse_args()

def render_ommatidia(sim):
    """Convert ommatidia readouts to a displayable RGB image."""
    ommatidia = sim.get_ommatidia_readouts(sim.fly.name)

    # Convert the two eyes to human-readable images
    left_eye = sim.fly.retina.hex_pxls_to_human_readable(
        ommatidia[0].max(-1), color_8bit=True # Corrected: ommatidia[0] for left eye
    )
    right_eye = sim.fly.retina.hex_pxls_to_human_readable(
        ommatidia[1].max(-1), color_8bit=True # Corrected: ommatidia[1] for right eye
    )
    vision_img = np.concatenate([left_eye, right_eye], axis=1)
    vision_img = np.stack([vision_img] * 3, axis=-1) # No difference but adds a 3rd argument (so the np.pad doesn't crash)

    return vision_img

def odor_to_drives(odor_intensities, attractive_gain=-500, aversive_gain=80): 
    n_sources = odor_intensities.shape[1]

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

def main():
    args = parse_args()

    # Game state and control setup
    game_state = GameState()

    sim = MiniprojectSimulation(
        level=args.level,
        seed=args.seed,
        hills=args.hills,
        grass=args.grass,
        wind=args.wind,
        dragonfly=args.dragonfly,
    )
    controller = TurningController(sim.timestep)
    odor_smooth = None
    odor_mode_enabled = False

    pygame.init()

    print("Getting controllers")
    if args.dont_use_pygame_rendering:
        from miniproject.interactive.controls_pynput import KeyboardControlPynput
        import cv2

        controls = KeyboardControlPynput(game_state)
        cv2.namedWindow(
            WINDOW_NAME,
        )
        cv2.setWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

        def get_controller_orders():
            pygame.event.pump()
            gain_left, gain_right, odor_mode_enabled = controls.get_actions(memory=False)
            return gain_left, gain_right

        def render(frame: np.ndarray):
            cv2.imshow(WINDOW_NAME, frame[..., ::-1])  # convert RGB to BGR for opencv
            cv2.waitKey(1)

    else:
        from miniproject.interactive import KeyboardControl

        controls = KeyboardControl(game_state)
        display_size = (1024, 1024 if args.render_fly_vision else 512)
        screen = pygame.display.set_mode(display_size)
        pygame.display.set_caption(WINDOW_NAME)

        def get_controller_orders():
            events = pygame.event.get()
            controls.process_events(events)
            keys_pressed = pygame.key.get_pressed()
            gain_left, gain_right, odor_mode_enabled, order_changing_banana_pos = controls.get_actions(keys_pressed, memory=False)
            return gain_left, gain_right, odor_mode_enabled, order_changing_banana_pos

        def render(frame: np.ndarray):
            frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            if frame_surface.get_size() != display_size:
                frame_surface = pygame.transform.smoothscale(
                    frame_surface, display_size
                )
            screen.blit(frame_surface, (0, 0))
            pygame.display.flip()

    step = 0
    while not game_state.get_quit():

        """ CONTROLLER GAINS """
        gain_left, gain_right, odor_mode_enabled, order_changing_banana_pos = get_controller_orders()


        """ ODOR DETECTION """
        olfaction = sim.get_olfaction(sim.fly.name)
        if odor_smooth is None:
            odor_smooth = olfaction
        else:
            odor_smooth = (1 - ALPHA) * odor_smooth + ALPHA * olfaction
        
        odor_drives = odor_to_drives(odor_smooth) * 3 if odor_mode_enabled else np.ones(2) 


        """ INSTRUCTIONS TO BODY """
        if game_state.get_reset():
            pass  # not implemented yet
        if game_state.get_quit():
            break
        drives = odor_drives * np.array([gain_left, gain_right])
        joint_angles, adhesion_signals = controller.step(drives)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion_signals)
        sim.step(change_pos_banana=order_changing_banana_pos)


        """ DISPLAYS """
        if sim.render_as_needed():
            frame = np.concatenate([frames[-1] for frames in sim.renderer.frames.values()], axis=-2)

            if args.render_fly_vision:
                fly_vision = render_ommatidia(sim)
                diff = frame.shape[1] - fly_vision.shape[1]
                fly_vision = np.pad(
                    fly_vision,
                    (
                        (0, 0),               
                        (diff // 2, diff - diff // 2), 
                        (0, 0),               
                    ),
                ) 
                frame_tot = np.vstack((fly_vision, frame))
                render(frame_tot)
            else:
                render(frame)


        step += 1
    controls.quit()
    pygame.quit()


if __name__ == "__main__":
    main()
