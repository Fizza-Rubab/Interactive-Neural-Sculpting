import time
import torch
from copy import deepcopy
from scipy.interpolate import CubicSpline
import sys, os
import numpy as np
sys.path.append( os.path.dirname( os.path.dirname(os.path.abspath(__file__) ) ) )
from ensdf import datasets, training, modules
from ensdf.raymarching import raymarch
from ensdf.rendering.camera import OrbitingCamera
from ensdf.modules import Siren
from ensdf.datasets import SDFEditingDataset
from ensdf.brushes import StrokeBrush
from ensdf.training import train_sdf
from ensdf.utils import get_cuda_if_available
from ensdf.aabb import AABB

def catmull_rom_spline(control_points, num_points=100):
    control_points = np.array(control_points)
    spline = CubicSpline(np.linspace(0, 1, len(control_points)), control_points, bc_type='clamped')
    t = np.linspace(0, 1, num_points)
    spline_points = spline(t)
    return spline, spline_points

def fixed_brush():
    control_points = np.array(sorted([(-1, 0.0), (0, 0.75), (1, 0.0)], key=lambda x: x[0]))
    return CubicSpline(control_points[:, 0], control_points[:, 1])

def fine_tune_model(model_path, epochs=50, num_ctrl=64, brush_profile=None, mode="tubular"):
    device = get_cuda_if_available()
    
    model = Siren.load(model_path)
    model.to(device)

    brush = StrokeBrush(
        brush_type="linear", 
        brush_profile=[(-1, 0.0), (0, 0.75), (1, 0.0)],
        template=fixed_brush(),
        radius=0.08,
        intensity=0.04,
        tubular=(mode == "tubular")
    )

    aabb = AABB([0., 0., 0.], [1., 1., 1.], device=device)
    camera = OrbitingCamera(np.deg2rad(60), (640, 360), np.deg2rad(0), np.deg2rad(90), 3.2)
    origins, directions = camera.generate_rays()
    origins = origins.to(device)
    directions = directions.to(device)

    traced_positions, traced_normals, traced_sdf, traced_hit = raymarch(
        model, aabb, origins, directions, num_iter=80
    )
    
    edit_dataset = SDFEditingDataset(model, device, brush, num_model_samples=1000)
    lr = 1e-4
    control_points = [(270, 238), (450, 238)]
    spline, int_points = catmull_rom_spline(control_points, num_ctrl)

    xs = [i for i, y in int_points]
    ys = [i for x, i in int_points]

    brush.set_curve(CubicSpline(np.linspace(0, 1, len(xs)), torch.rand((len(xs), 3)).cpu().numpy(), bc_type='clamped'))  # Replace random positions with actual data
    brush.set_interaction(
                        traced_positions[xs, ys],
                        traced_normals[xs, ys],
                        traced_sdf[xs, ys]
                        )
    prev_model = deepcopy(model)

    start_time = time.time()
    trials = 5
    for _ in range(trials):
        train_sdf(
            model=model,
            surface_dataset=edit_dataset,
            lr=lr,
            epochs=epochs,
            device=device,
            regularization_samples=10_000,
            freeze_initial=False
        )
    end_time = time.time()

    elapsed_time = (end_time - start_time) / trials
    print(f"Fine-tuning (Mode: {'Tubular' if mode == 'tubular' else 'Non-Tubular'}) took {elapsed_time:.6f} seconds")

    return model

if __name__ == "__main__":
    model_path = "../pretrained_models/sphere.pth"  
    fine_tuned_model_tubular = fine_tune_model(model_path, epochs=50, num_ctrl=64, mode="tubular")
    # fine_tuned_model_non_tubular = fine_tune_model(model_path, epochs=50, mode="tubular")