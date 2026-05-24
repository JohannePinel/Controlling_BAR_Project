import numpy as np
import mediapy
from miniproject import MiniprojectSimulation
from flygym.vision.retina import Retina

data   = np.load("flat_grass_1779643523.npz", allow_pickle=False)
inputs = data["inputs"]  # (N, 5, 2, 721, 2)
gains  = data["gains"]   # (N, 2)
modes  = data["modes"]   # (N,)
steps  = data["steps"]   # (N,)

print(f"N samples : {len(inputs)}")
print(f"Modes     : {np.unique(modes, return_counts=True)}")
print(f"Gains min/max : {gains.min():.2f} / {gains.max():.2f}")
print(f"NaN check : {np.isnan(inputs).any()}")  # doit être False


frames = []
retina = Retina()
for sample in inputs:
    left  = retina.hex_pxls_to_human_readable(sample[4, 0, :, :], color_8bit=True)
    right = retina.hex_pxls_to_human_readable(sample[4, 1, :, :], color_8bit=True)
    print(f"left shape: {left.shape}, right shape: {right.shape}")
    frame = np.concatenate([left, right], axis=1)  # côte à côte
    frame_one_channel = frame[..., 0]
    frames.append(frame_one_channel)

# Affiche les 5 steps de la fenêtre comme une vidéo
mediapy.show_video(frames, fps=2, title=f"cul")