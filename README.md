# Interactive Stroke-based Neural SDF Sculpting

This repository provides the source code for the paper **Interactive Stroke-based Neural SDF Sculpting**. The framework allows intuitive, real-time stroke-based editing of 3D Signed Distance Functions (SDFs), enabling smooth, customizable deformations.

## Quick Start
To run the interactive editing framework with a pretrained model, execute:

```
python viewer/stroke.py pretrained_models/sphere.pth
```

### Usage Instructions
- Point Edit: Click anywhere on the screen to perform a point edit.  
- Stroke Editing:  
  2. Click on control points along the desired stroke path on the shape in SDF editor.  
  3. Press `S` to finalize the stroke and view the result.  
- Other Controls:  
  - Change Modulation: Press `Ctrl + M` to select from the  modulation functions.  Type one from `identity, central, linear, sinusoidal, inverse_central`.
  - Navigation: Use arrow keys to move around the viewport.  
  - Change brush radius: Use up/down keys to change brush radius. 
  - Change brush intensity: Use left/right keys to change brush intensity.  
  - Custom Brush Profiles: Press `B` to switch to a different custom brush profile.  
  - Undo Edit: Press `Ctrl + Z` to undo the previous edit. Only 1 edit can be reversed currently. 
  - Save Snapshot: Press `Enter` to save the current view and provide file name e.g `bunny.png`.  
  - Save Model: Press `Ctrl + Enter` to save the current model and provide file name `edited_model.pth`.  

## Code Structure
The code structure is adapted from the 3DNS framework. Below is an overview of the key components:

- `ensdf/`: Contains core library code for neural networks, loss functions, and utilities.  
- `scripts/`: Includes scripts for experiments such as timing analysis and mesh comparison.  
- `viewer/`: Contains the interactive editor, specifically `viewer/stroke.py`.  

To train a neural SDF from a mesh, run

```
"python scripts/train_mesh_sdf.py --mesh_path <mesh_path> --model_dir <model_dir_path> --num_epochs 300000
```

To train an analytic sphere or torus run  `scripts/train_sphere_sdf.py` and `scripts/train_torus_sdf.py`



### Acknowledgments
This code builds upon the following open-source libraries:  
1. [3DNS](https://github.com/pettza/3DNS)  
2. [SIREN](https://github.com/vsitzmann/siren)  
3. [DeepSDF](https://github.com/facebookresearch/DeepSDF)  
4. [NDF](https://github.com/jchibane/ndf)  

### Citation
Please cite our paper if you refer to our results or use the method or code in your own work:

    

      @inproceedings{10.2312:hpg.20251169,
      booktitle = {High-Performance Graphics - Symposium Papers},
      title = {{Interactive Stroke-based Neural SDF Sculpting}},
      author = {Rubab, Fizza and Tong, Yiying},
      year = {2025},
      publisher = {The Eurographics Association},
      ISSN = {2079-8687},
      ISBN = {978-3-03868-291-2},
      DOI = {10.2312/hpg.20251169}
      }
        
   