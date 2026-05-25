from tqdm import trange
from flygym.compose import ActuatorType
import importlib
import miniproject.simulation
importlib.reload(miniproject.simulation)
from miniproject.simulation import MiniprojectSimulation
from controller_ML import Controller
import mediapy

sim = MiniprojectSimulation(hills=False, grass=True, wind=False, dragonfly=False)
controller = Controller(sim)

for _ in trange(100000):
    joint_angles, adhesion = controller.step(sim)
    sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
    sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
    sim.step()
    sim.render_as_needed()



frames = list(sim.renderer.frames.values())[0]
mediapy.write_video("output.mp4", frames, fps=30)
print("Vidéo sauvegardée : output.mp4")