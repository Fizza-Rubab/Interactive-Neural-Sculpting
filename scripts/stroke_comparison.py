import sys
import os
import copy

import torch
import torch.nn.functional as F
import numpy as np
import trimesh
import shutil
import random
from options import create_edit_parser
sys.path.append( os.path.dirname( os.path.dirname(os.path.abspath(__file__) ) ) )
from ensdf import datasets, training, modules
from ensdf.aabb import AABB
from ensdf.geoutils import linear_fall
from ensdf.rendering.camera import OrbitingCamera
from ensdf.raymarching import raymarch_single_ray, raymarch
from ensdf.meshing import marching_cubes
from ensdf.brushes import SimpleBrush, StrokeBrush
from ensdf.geoutils import normalize_trimesh, cubic_fall, quintic_fall
from ensdf.meshing import marching_cubes
from ensdf.utils import get_cuda_if_available, simplify_trimesh, size_of_trimesh
from ensdf.metrics import chamfer
from ensdf.sampling.sdf import SDFSampler
from scipy.interpolate import CubicSpline, BSpline

def gather_samples_in_interaction(brush, sample_fn, n_samples):
    gathered_samples = 0
    samples = []
    i = 0
    while gathered_samples < n_samples:
        iter_samples = sample_fn()
        cond = brush.inside_interaction(iter_samples)
        samples.append(iter_samples[cond])
        gathered_samples += samples[-1].shape[0]
        i += 1
        if i > 50 and gathered_samples == 0:
            break 

    samples = torch.concat(samples)[:n_samples]
    return samples

def catmull_rom_spline(control_points, num_points=100):
    control_points = np.array(control_points)
    print(control_points)
    spline = CubicSpline(np.linspace(0, 1, len(control_points)), control_points, bc_type='clamped')
    t = np.linspace(0, 1, num_points)
    spline_points = spline(t)
    return spline, spline_points



def main():
    arg_parser, arg_groups = create_edit_parser()

    # Mesh options
    mesh_group = arg_parser.add_argument_group('Mesh options')
    mesh_group.add_argument('--mesh_path', type=str,
                            help='Path to the mesh that the network was trained on.')
    
    # Comparison options
    comparison_group = arg_parser.add_argument_group('Mesh options')
    comparison_group.add_argument('--chamfer_samples', type=float, default=100_000,
                                  help='The number of points to be used for chamfer distance')
    comparison_group.add_argument('--average', type=int, default=None,
                                  help='If specified, the average of multiple random edits \
                                        will be computed')
    comparison_group.add_argument('--num_edits', type=int, default=None,
                                  help='If specified, the edit from interaction options \
                                        will be ignored and instead the given number of points \
                                        will be randomly chosen on the surface. This will happen \
                                        per iteration if --average is also specified.')

    options = arg_parser.parse_args()

    os.makedirs(options.model_dir, exist_ok=True)
    camera = OrbitingCamera(np.deg2rad(60), (640, 360), np.deg2rad(0), np.deg2rad(0), 3.2)
    device = get_cuda_if_available()
    model = modules.Siren.load(options.model_path)
    model.to(device)
    model_size = model.get_num_bytes()

    if options.mesh_path:
        print(f'Reading mesh from {options.mesh_path}...')
        original_mesh = trimesh.load_mesh(options.mesh_path)
        normalize_trimesh(original_mesh, border=0.15)
        print(f'Done')
    else:
        print('Creating mesh...')
        original_mesh = marching_cubes(model)
        print('Done')

    print('Simplifying mesh...')
    original_num_faces = original_mesh.faces.shape[0]
    original_size = size_of_trimesh(original_mesh)
    simple_num_faces, simple_mesh, simple_size = simplify_trimesh(original_mesh, model_size)
    print('Done')

    print(f'Size of model: {model_size}')
    print(f'Number of faces and size of original mesh: {original_num_faces} {original_size}')
    print(f'Number of faces and size of simple mesh: {simple_num_faces} {simple_size}')

    brush_profile = np.array([(-1.0, 0.), (0.0, 1.0), (1.0, 0.)])
    brush_template = CubicSpline(brush_profile[:, 0], brush_profile[:, 1])
    # brush_template = quintic_fall
    brush = StrokeBrush(
            brush_type=options.brush_type,
            brush_profile=brush_profile,
            template=brush_template,
            radius=0.08,
            intensity=0.06,
            tubular=False,
    )
    brush2 = StrokeBrush(
            brush_type=options.brush_type,
            brush_profile=brush_profile,
            template=brush_template,
            radius=0.08,
            intensity=0.06,
            tubular=True,
    )
    ct = 8
    random_edits = options.average or options.num_edits
    num_edits = options.num_edits or 1

    edit_iterations = options.average or 1
    model_total_dists = np.empty(edit_iterations)
    model_total_dists2 = np.empty(edit_iterations)
    model_inter_dists = np.empty((edit_iterations, num_edits))
    model_inter_dists2 = np.empty((edit_iterations, num_edits))
    simple_mesh_total_dists = np.empty(edit_iterations)
    simple_mesh_inter_dists = np.empty((edit_iterations, num_edits))

    it = 0
    while it<edit_iterations:
        print(f"Iteration: {it}")
        model_copy = copy.deepcopy(model)
        model_copy2 = copy.deepcopy(model)
        original_mesh_edited = copy.deepcopy(original_mesh)
        simple_mesh_edited = copy.deepcopy(simple_mesh)

        model_sampler = SDFSampler(model_copy, device, options.chamfer_samples, burnout_iters=100)
        model_sampler2 = SDFSampler(model_copy2, device, options.chamfer_samples, burnout_iters=100)
        
        # Edit model
        dataset = datasets.SDFEditingDataset(
            model_copy, device, brush,
            num_model_samples=10000,
            interaction_samples_factor=options.interaction_samples_factor
        )
        dataset2 = datasets.SDFEditingDataset(
            model_copy2, device, brush2,
            num_model_samples=10000,
            interaction_samples_factor=options.interaction_samples_factor
        )

        print("it is not a random edit hehe")
        aabb = AABB([0., 0., 0.], [1., 1., 1.], device=device)
        origins, directions = camera.generate_rays()
        origins = origins.to(device)
        directions = directions.to(device)

        inter_points, inter_normals, inter_sdfs, ray_hit = raymarch(
            model, aabb, origins, directions, num_iter=80
        )

        xc, yc = random.randint(220, 320), random.randint(140, 180) 
        ls = [(xc, yc), (xc + random.randint(0, 60), yc + random.randint(0, 60))]
        print("ls", ls)
        # ls = [(253, 110), (279, 96), (287, 78), (322, 63), (363, 60), (361, 96), (386, 110)]
        # ls = [(222, 160), (219, 48)]
        # ls = [(406, 229), (226, 229)]
        spline, int_points = catmull_rom_spline(ls, ct)
        # spline, int_points = catmull_rom_spline([(206, 221), (202, 330)], ct)
        xs = [i for i, y in int_points] 
        ys = [i for x, i in int_points]


        inter_points = inter_points[xs, ys]
        inter_normals = inter_normals[xs, ys]
        inter_sdfs = inter_sdfs[xs, ys]

        brush.set_curve(CubicSpline(np.linspace(0, 1, len(xs)), inter_points.cpu().numpy(), bc_type='clamped'))
        brush.set_interaction(
        inter_points,
        inter_normals,
        inter_sdfs
        )
        training.train_sdf(
            model=model_copy, surface_dataset=dataset, epochs=100, lr=options.lr,
            epochs_til_checkpoint=options.num_epochs, pretrain_epochs=0,
            regularization_samples=options.regularization_samples,
            include_empty_space_loss=not options.no_empty_space,
            ewc=options.ewc, device=device,
            freeze_initial=False
        )   
        try:
            dataset.update_model(model_copy, sampler_iters=20)
        except:
            print("Skipping this iteration")
            continue
        model_sampler.burnout(20)
        print("Model 1 trained")


        brush2.set_curve(CubicSpline(np.linspace(0, 1, len(xs)), inter_points.cpu().numpy(), bc_type='clamped'))
        brush2.set_interaction(
        inter_points,
        inter_normals,
        inter_sdfs
        )
        training.train_sdf(
            model=model_copy2, surface_dataset=dataset2, epochs=100, lr=options.lr,
            epochs_til_checkpoint=options.num_epochs, pretrain_epochs=0,
            regularization_samples=options.regularization_samples,
            include_empty_space_loss=not options.no_empty_space,
            ewc=options.ewc, device=device,
            freeze_initial=False
        )   
        try:
            dataset2.update_model(model_copy2, sampler_iters=20)
        except:
            print("Skipping this iteration")
            continue
        model_sampler2.burnout(20)
        print("Model 2 trained")

        original_mesh_edited = brush2.deform_mesh(original_mesh_edited)
        simple_mesh_edited = brush2.deform_mesh(simple_mesh_edited)

        if not options.average:
            # Save meshes and model
            original_mesh_edtied_path = os.path.join(options.model_dir, f'original_mesh_edited_{ct}.obj')
            simple_mesh_edited_path = os.path.join(options.model_dir, f'simple_mesh_edited_{ct}.obj')
            model_path = os.path.join(options.model_dir, f'model_tubular=False_{ct}.pth')
            model_path2 = os.path.join(options.model_dir, f'model_tubular=True_{ct}.pth')
            model_mesh_edited = marching_cubes(model_copy, 256, max_batch=128**3)
            model_mesh_edited2 = marching_cubes(model_copy2, 256, max_batch=128**3)
            model_mesh_edited.export(os.path.join(options.model_dir, f"model_tubular=False_{ct}.obj"))
            model_mesh_edited2.export(os.path.join(options.model_dir, f"model_tubular=True_{ct}.obj"))

            original_mesh_edited.export(original_mesh_edtied_path)
            simple_mesh_edited.export(simple_mesh_edited_path)
            model_copy.save(model_path)
            model_copy2.save(model_path2)


        original_mesh_edited_dataset = datasets.MeshDataset(
            original_mesh_edited, options.chamfer_samples, device, normalize=False
        )
        simple_mesh_edited_dataset = datasets.MeshDataset(
            simple_mesh_edited, options.chamfer_samples, device, normalize=False
        )
        print("datasets created")

        # Chamfer distance over interaction
        original_mesh_editied_inter_pc = gather_samples_in_interaction(
            brush, lambda: original_mesh_edited_dataset.sample()['points'], options.chamfer_samples
        ).cpu().numpy()
        
        if original_mesh_editied_inter_pc.shape[0] == 0:
            print("Skipping this iteration, not enough chamfer samples")
            continue


        print("mesh gathered")
        simple_mesh_edited_inter_pc = gather_samples_in_interaction(
            brush, lambda: simple_mesh_edited_dataset.sample()['points'], options.chamfer_samples
        ).cpu().numpy()
        print("simple gathered")

        model_inter_pc = gather_samples_in_interaction(
            brush, lambda: next(model_sampler)['points'], options.chamfer_samples
        ).cpu().numpy()
        print("model inter gathered")

        model_inter_pc2 = gather_samples_in_interaction(
            brush, lambda: next(model_sampler2)['points'], options.chamfer_samples
        ).cpu().numpy()
        print("model2 gathered")


        model_inter_dist = chamfer(original_mesh_editied_inter_pc, model_inter_pc)
        model_inter_dist2 = chamfer(original_mesh_editied_inter_pc, model_inter_pc2)
        simple_mesh_inter_dist = chamfer(original_mesh_editied_inter_pc, simple_mesh_edited_inter_pc)
        model_inter_dists[it][0] = model_inter_dist
        model_inter_dists2[it][0] = model_inter_dist2
        simple_mesh_inter_dists[it][0] = simple_mesh_inter_dist

        original_mesh_edited_dataset = datasets.MeshDataset(
            original_mesh_edited, options.chamfer_samples, device, normalize=False
        )
        simple_mesh_edited_dataset = datasets.MeshDataset(
            simple_mesh_edited, options.chamfer_samples, device, normalize=False
        )
    
        # Chamfer distance over entire surface
        original_mesh_editied_total_pc = original_mesh_edited_dataset.sample()['points'].cpu().numpy()
        simple_mesh_edited_total_pc = simple_mesh_edited_dataset.sample()['points'].cpu().numpy()
        model_total_pc = next(model_sampler)['points'].cpu().numpy()
        model_total_pc2 = next(model_sampler2)['points'].cpu().numpy()


        model_total_dist = chamfer(original_mesh_editied_total_pc, model_total_pc)
        model_total_dist2 = chamfer(original_mesh_editied_total_pc, model_total_pc2)
        simple_mesh_total_dist = chamfer(original_mesh_editied_total_pc, simple_mesh_edited_total_pc)
        model_total_dists[it] = model_total_dist
        model_total_dists2[it] = model_total_dist2
        simple_mesh_total_dists[it] = simple_mesh_total_dist

        print("chamfer computed")
        it +=1

    def write_distances(f, dists):
        if dists.size > 1:
            print("dists", dists)
            f.write(f'\tMean: {dists.mean(axis=0)}\n')
            f.write(f'\tStd: {dists.std(axis=0)}\n')
        else:
            f.write(f'\t{dists.item(0)}\n')
    
    results_filename = os.path.join(options.model_dir, 'chamfer_distances.txt')
    with open(results_filename, 'w') as f:
        f.write('- Chamfer distance over entire surface\n')
        f.write(f'Tubular: False, Number of samples: {model_total_dists.size}\n')
        f.write('Model - Original Mesh - :\n')
        write_distances(f, model_total_dists)
        f.write(f'Tubular: True, Number of samples: {model_total_dists2.size}\n')
        f.write('Model - Original Mesh - :\n')
        write_distances(f, model_total_dists2)
        f.write('Simple Mesh - Original Mesh:\n')
        write_distances(f, simple_mesh_total_dists)

        f.write('\n')

        f.write('- Chamfer distance over interaction\n')
        f.write('Model - Original Mesh:\n')
        f.write(f'Tubular: False, Number of samples: {model_inter_dists.size}\n')
        write_distances(f, model_inter_dists)
        f.write(f'Tubular: True, Number of samples: {model_inter_dists2.size}\n')
        write_distances(f, model_inter_dists2)
        f.write('Simple Mesh - Original Mesh:\n')
        write_distances(f, simple_mesh_inter_dists)

    with open(results_filename) as f:
        print(f.read())


if __name__ == '__main__':
    main()