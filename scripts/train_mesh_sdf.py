import sys
import os

from options import create_parser, add_training_options, add_model_options

sys.path.append( os.path.dirname( os.path.dirname(os.path.abspath(__file__) ) ) )
from ensdf import datasets, training, modules
from ensdf.meshing import marching_cubes
from ensdf.utils import get_cuda_if_available


def main():
    arg_parser = create_parser()
    training_group = add_training_options(arg_parser)
    model_group = add_model_options(arg_parser)

    # Dataset options
    dataset_group = arg_parser.add_argument_group('Dataset options')
    dataset_group.add_argument('--surface_samples', type=int, default=120_000,
                               help='Number of on surface samples per training iteration.')
    dataset_group.add_argument('--mesh_path', type=str, required=True,
                               help='Path to the mesh file.')
    model_group.add_argument("--wire", action='store_true')
    model_group.add_argument("--omega", default=10)
    model_group.add_argument("--scale", default=5)


    options = arg_parser.parse_args()

    device = get_cuda_if_available()
    dataset = datasets.MeshDataset(
        mesh_or_path=options.mesh_path,
        num_samples=options.surface_samples,
        device=device
    )
    if not options.wire:
        model = modules.Siren(
            in_features=3, hidden_features=128,
            hidden_layers=2, out_features=1,
            weight_norm=options.weight_norm,
            first_omega_0=30, outermost_linear=True
        )
    else:
        model = modules.INR(in_features=3, hidden_features=128, 
                    hidden_layers=1, 
                    out_features=1, outermost_linear=True,
                    first_omega_0=int(options.omega), hidden_omega_0=int(options.omega), scale=int(options.scale),
                    pos_encode=False, sidelength=512, fn_samples=None,
                    use_nyquist=True
                    )
    print("Model", model)
    training.train_sdf(
        model=model, surface_dataset=dataset, epochs=options.num_epochs, lr=options.lr,
        epochs_til_checkpoint=options.epochs_til_ckpt, pretrain_epochs=options.pretrain_epochs,
        regularization_samples=options.regularization_samples,
        model_dir=options.model_dir, device=device
    )

    mesh = marching_cubes(model)
    mesh_dir = os.path.join(options.model_dir, 'mesh')
    os.mkdir(mesh_dir)
    mesh.export(os.path.join(mesh_dir, 'mesh.ply'))


if __name__ == '__main__':
    main()
