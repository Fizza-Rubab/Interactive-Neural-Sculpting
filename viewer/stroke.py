import sys
import os
from copy import deepcopy
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pyglet
from pyglet.window import key, mouse
from OpenGL.GL import *
import time
from viewer_utils import ScreenTexture, BrushRenderer
import matplotlib.pyplot as plt
sys.path.append( os.path.dirname( os.path.dirname(os.path.abspath(__file__) ) ) )
import ensdf.modules as modules
from ensdf.brushes import SimpleBrush, StrokeBrush
from ensdf.aabb import AABB
from ensdf.raymarching import raymarch
from ensdf.meshing import marching_cubes
from ensdf.rendering.camera import OrbitingCamera
from ensdf.rendering.shading import phong_shading, shade_normals
from ensdf.datasets import SDFEditingDataset
from ensdf.training import train_sdf
from ensdf.utils import get_cuda_if_available
from scipy.interpolate import CubicSpline, BSpline
from matplotlib.backend_bases import MouseButton
import gc
import matplotlib as mpl
import matplotlib.font_manager as font_manager

mpl.rcParams['font.family'] = 'serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif'] = cmfont.get_name()
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False

def catmull_rom_spline(control_points, num_points=100):
    control_points = np.array(control_points)
    print(control_points)
    spline = CubicSpline(np.linspace(0, 1, len(control_points)), control_points, bc_type='clamped')
    t = np.linspace(0, 1, num_points)
    spline_points = spline(t)
    return spline, spline_points


class ENSDFWindow(pyglet.window.Window):
    def __init__(self, resolution=(720, 480), epochs_per_edit=50, brush_profile=None, mode="tubular"):
        super().__init__(caption="SDF Viewer", width=resolution[0], height=resolution[1])

        self.resolution = resolution
        self.epochs_per_edit = epochs_per_edit
        self.edit_dict = {"edits":[]}
        if not os.path.exists("edits"):
            os.mkdir("edits")
        self.edit_file = f"edits/edit_{time.strftime('%Y%m%d-%H%M%S')}.json"
        script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
        shader_dir = os.path.join(script_dir, 'shaders')

        self.screen_tex = ScreenTexture(
            resolution,
            os.path.join(shader_dir, 'screen_texture_vert.vs'),
            os.path.join(shader_dir, 'screen_texture_frag.fs')
        )
                

        self.device = get_cuda_if_available()
        self.model = modules.Siren.load(sys.argv[1])
        self.model.to(self.device)
        self.prev_model = self.model
        self.weight_stack = []
        self.aabb = AABB([0., 0., 0.], [1., 1., 1.], device=self.device)
        self.camera = OrbitingCamera(np.deg2rad(50), self.resolution, np.deg2rad(0), np.deg2rad(90), 2.2)

        self.background_color = torch.tensor([1.0, 1.0, 1.0], device=self.device).view(1, 1, 3)
        self.light_colors = torch.tensor([1., 1., 1.], device=self.device)

        self.brush_types = ['linear', 'cubic', 'quintic', 'exp', "2D"]
        self.brush_type_idx = 0 
        brush_template = CubicSpline(brush_profile[:, 0], brush_profile[:, 1])
        self.brush = StrokeBrush(
            brush_type=self.brush_types[self.brush_type_idx],
            brush_profile=brush_profile,
            template=brush_template,
            radius=0.05,
            intensity=0.03,
            tubular=(mode=="tubular")
            
        )
        
        self.ct = 99
        self.edit_dataset = SDFEditingDataset(
            self.model, self.device, self.brush,
            num_model_samples=10000
        )
        self.lr = 1e-4
        
        self.brush_renderer = BrushRenderer(
            self.camera, self.brush,
            os.path.join(shader_dir, 'brush_vert.vs'),
            os.path.join(shader_dir, 'brush_frag.fs')
        )
        
        self.mouse_pos = (0, 0)
        self.retrace = True

        self.drag = False
        self.stroke_path = True
        self.control_points = []
        self.click = None
        self.stroke_points = []

        print(f"Curve Tracing: {self.stroke_path}")

    def print_brush(self):
        print(f'Radius: {self.brush.radius:0.3f} | ', sep='')
        print(f'Intensity: {self.brush.intensity:0.3f} | ', sep='')
        print(f'Type: {self.brush.brush_type}')

    def on_draw(self):
        self.clear()

        if self.retrace:
            origins, directions = self.camera.generate_rays()
            origins = origins.to(self.device)
            directions = directions.to(self.device)

            self.traced_positions, self.traced_normals, self.traced_sdf, self.traced_hit = raymarch(
                self.model, self.aabb, origins, directions, num_iter=60
            )
            F.normalize(self.traced_normals, dim=-1, out=self.traced_normals)
            
            eye_pos = self.camera.position.to(self.device)
            light_pos = eye_pos 
            colors = phong_shading(
                self.traced_positions, self.traced_normals, light_pos, self.light_colors, eye_pos
            )

            image = torch.where(self.traced_hit, colors, self.background_color).transpose_(0, 1)
            torch.clamp(image, min=0., max=1., out=image)

            tex_data = image.detach().cpu().numpy()

            self.screen_tex.update_texture(tex_data)
            
            self.retrace = False
        
        self.screen_tex.render()

        x, y = self.mouse_pos
        if self.valid_interaction(x, y):
            self.brush.set_interaction(
                self.traced_positions[x, y:y+1],
                self.traced_normals[x, y:y+1],
                self.traced_sdf[x, y:y+1]
            )
            self.brush_renderer.render()

       
    def valid_interaction(self, mouse_x, mouse_y):
        cond = (
            0 <= mouse_x < self.resolution[0] and
            0 <= mouse_y < self.resolution[1] and
            self.traced_hit[mouse_x, mouse_y]
        )
        return cond

    def on_key_press(self, symbol, modifiers):
        if modifiers & key.MOD_ALT:
            if symbol == key.UP:
                self.brush_type_idx += 1
                self.brush_type_idx %= len(self.brush_types)
                self.brush.brush_type = self.brush_types[self.brush_type_idx]
                self.print_brush()
            elif symbol == key.DOWN:
                self.brush_type_idx -= 1
                self.brush_type_idx %= len(self.brush_types)
                self.brush.brush_type = self.brush_types[self.brush_type_idx]
                self.print_brush()
        elif modifiers & key.MOD_CTRL:
            if symbol == key._1:
                self.brush.profile = np.array([(-1, 0.0),(0.0, 1), (1.0, 0.0)])
                self.brush.template = CubicSpline(self.brush.profile[:, 0], self.brush.profile[:, 1])
            elif symbol == key._2:
                self.brush.profile = np.array([[-1, 0], [-0.5, 1], [0.5, -1], [1, 0.0]])
                self.brush.template = CubicSpline(self.brush.profile[:, 0], self.brush.profile[:, 1])
            elif symbol == key._3:
                self.brush.profile = np.array([(-1, 0.0), (-0.6, -0.25), (0.0, 1), (0.6, -0.25), (1.0, 0.0)])
                self.brush.template = CubicSpline(self.brush.profile[:, 0], self.brush.profile[:, 1])
            elif symbol == key._4:
                self.brush.profile = np.array([(-1, 0.0), (-0.75, 1), (-0.25, -1), (0.25, 1), (0.75, -1), (1, 0.0)])
                self.brush.template = CubicSpline(self.brush.profile[:, 0], self.brush.profile[:, 1])
            elif symbol == key.UP:
                self.brush.radius += 0.005
                self.print_brush()
            elif symbol == key.DOWN:
                self.brush.radius -= 0.005
                self.print_brush()
            elif symbol == key.LEFT:
                self.brush.intensity -= 0.005
                self.print_brush()
            elif symbol == key.RIGHT:
                self.brush.intensity += 0.005
                self.print_brush()
            elif symbol == key.ENTER:
                model_path = input('Model path: ').strip()
                self.model.save(model_path)
            elif symbol == key.O:
                mesh_path = input('Mesh path: ').strip()
                mesh = marching_cubes(self.model, 256, max_batch=128**3)
                mesh.export(mesh_path)
            elif symbol == key.M:
                mode = input('Modulation (identity, central, linear, sinusoidal, inverse_central): ').strip()
                self.brush.mode = mode
            elif symbol == key.Z:
                if self.model is not self.prev_model and self.edit_dict["edits"]:
                    self.model = self.prev_model
                    self.edit_dataset.update_model(self.model, sampler_iters=10)
                    self.edit_dict["edits"].pop()
                    json.dump(self.edit_dict, open(self.edit_file, "w"), indent=True)
                    self.retrace = True
        else:    
            if symbol == key.UP:
                self.camera.theta -= np.pi / 10
                print(f"Camera - theta: {np.rad2deg(self.camera.theta)}, phi: {np.rad2deg(self.camera.phi)}")
                self.retrace = True
            elif symbol == key.DOWN:
                self.camera.theta += np.pi / 10
                print(f"Camera - theta: {np.rad2deg(self.camera.theta)}, phi: {np.rad2deg(self.camera.phi)}")
                self.retrace = True
            elif symbol == key.LEFT:
                self.camera.phi -= np.pi / 10
                print(f"Camera - theta: {np.rad2deg(self.camera.theta)}, phi: {np.rad2deg(self.camera.phi)}")
                self.retrace = True
            elif symbol == key.RIGHT:
                self.camera.phi += np.pi / 10
                print(f"Camera - theta: {np.rad2deg(self.camera.theta)}, phi: {np.rad2deg(self.camera.phi)}")
                self.retrace = True
            elif symbol == key.ENTER:
                img = np.flipud(self.screen_tex.tex_data)
                img = (img * 255).astype(np.uint8)
                pil_img = Image.fromarray(img).convert('RGBA')
                filename = input('Image filename: ').strip()
                pil_img.save(filename)
            elif symbol == key.M:
                self.brush.tubular = not self.brush.tubular
                print("Tubular", self.brush.tubular)
            elif symbol == key.B:
                self.brush.brush_profile = create_brush_profile()
                self.brush.template = CubicSpline(self.brush.brush_profile[:, 0], self.brush.brush_profile[:, 1])
            elif symbol == key.S:
                if self.stroke_path and (len(self.control_points)>1):
                    self.stroke_path = not(self.stroke_path)

                    spline, int_points = catmull_rom_spline(self.control_points, self.ct)
                    xs = [i for i, y in int_points]
                    ys = [i for x, i in int_points]
                    
                    self.brush.set_curve(CubicSpline(np.linspace(0, 1, len(xs)), self.traced_positions[xs, ys].cpu().numpy(), bc_type='clamped'))
                    self.brush.set_interaction(
                            self.traced_positions[xs, ys],
                            self.traced_normals[xs, ys],
                            self.traced_sdf[xs, ys]
                        )
                    self.prev_model = deepcopy(self.model)
                    st = time.time()
                    train_sdf(
                        model=self.model,
                        surface_dataset=self.edit_dataset,
                        lr=self.lr,
                        epochs=100,
                        device=self.device,
                        regularization_samples=0,
                        freeze_initial=False
                    )
                    et = time.time()
                    edit = {"intensity": np.round(self.brush.intensity, 3), "radius": np.round(self.brush.radius, 3), "stroke": self.control_points, "profile": self.brush.brush_profile.tolist(), "time": et-st}
                    print("Edit:", edit)
                    self.edit_dict["edits"].append(edit)
                    json.dump(self.edit_dict, open(self.edit_file, "w"), indent=2)
                    self.edit_dataset.update_model(self.model)
                    self.retrace = True
                    new_weights = deepcopy(self.model).state_dict()
                    self.weight_stack.append(new_weights)
                    self.control_points = []
                self.stroke_path = not(self.stroke_path)
                print(f"Curve Tracing: {self.stroke_path}")
    
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if scroll_y > 0:
            self.camera.radius *= 0.9
            self.retrace = True
        elif scroll_y < 0:
            self.camera.radius *= 1/0.9
            self.retrace = True
    
    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_pos = (x, y)
    
    def on_mouse_press(self, x, y, button, modifiers):
        if button == mouse.LEFT and self.valid_interaction(x, y):
            self.click = (x,y,)
            print("Mouse clicked at", x, y)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & mouse.LEFT and self.valid_interaction(x, y):
            if self.click:
                self.drag = True
                self.stroke_points.append((x, y))


    def on_mouse_release(self, x, y, button, modifiers):
        if button == mouse.LEFT and not self.drag and self.click:
            if self.valid_interaction(x, y) and not self.stroke_path:
                x, y = [x], [y]
                self.brush.set_interaction(
                    self.traced_positions[x, y],
                    self.traced_normals[x, y],
                    self.traced_sdf[x, y]
                )
                self.prev_model = deepcopy(self.model)
                st = time.time()
                train_sdf(
                    model=self.model,
                    surface_dataset=self.edit_dataset,
                    lr=self.lr,
                    epochs=self.epochs_per_edit,
                    device=self.device,
                    regularization_samples=120_000,
                    freeze_initial=True
                )
                et = time.time()
                edit = {"intensity": self.brush.intensity, "radius": self.brush.radius, "stroke": [(x[0], y[0])], "profile": self.brush.brush_profile.tolist(), "time": et-st}
                self.edit_dict["edits"].append(edit)
                print("Edit:", edit)
                json.dump(self.edit_dict, open(self.edit_file, "w"), indent=2)
                self.edit_dataset.update_model(self.model)
                self.retrace = True
                new_weights = deepcopy(self.model).state_dict()
                self.weight_stack.append(new_weights)
                self.control_points = []
                print(f"You clicked here: {self.click}")

            elif self.valid_interaction(x, y) and self.stroke_path:
                self.control_points.append((x, y))
        else:
            pass


def create_brush_profile(domain=[-1.0, 1.0]):
    control_points = [(domain[0], 0.0), (domain[1], 0.0)]
    control_points_arr = np.array(control_points)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    fig.set_dpi(100)
    ax.scatter(control_points_arr[:, 0], control_points_arr[:, 1], color='red', label="Control Points")
    y_vals, y_vals = None, None
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            control_points.append((event.xdata, event.ydata))
            plt.scatter(event.xdata, event.ydata, color='red')
            ax.set_aspect('equal')
            plt.draw()

    def onkey(event):
        global x_vals, y_vals
        if event.key == 'enter':
            if len(control_points) < 2:
                print("At least 2 points are needed to create a profile.")
                return
            
            control_points_sorted = sorted(control_points, key=lambda x: x[0])
            print("Final", control_points)
            control_points_arr = np.array(control_points_sorted)
            
            spline = CubicSpline(control_points_arr[:, 0], control_points_arr[:, 1])
            x_vals = np.linspace(domain[0], domain[1], 100)
            y_vals = spline(x_vals)
            ax.clear()
            ax.set_xlim(domain[0], domain[1])
            ax.set_ylim(domain[0], domain[1])
            ax.set_title("Draw your brush profile")
            ax.plot(x_vals, y_vals, label="Fitted Spline")
            ax.scatter(control_points_arr[:, 0], control_points_arr[:, 1], color='red', label="Control Points")
            ax.set_aspect('equal')
            ax.legend()
            plt.draw()

    ax.set_title("Draw your own brush profile")
    ax.set_xlim(domain[0], domain[1])
    ax.set_ylim(domain[0], domain[1])
    ax.set_aspect('equal')

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)

    plt.show(block=True)
    plt.close(fig)
    plt.close('all')
    gc.collect()  

    if len(control_points) < 2:
        raise ValueError("At least 2 points are required to create a brush profile.")

    control_points_sorted = sorted(control_points, key=lambda x: x[0])
    control_points_sorted = np.array(control_points_sorted)

    spline = CubicSpline(control_points_sorted[:, 0], control_points_sorted[:, 1])
    x_vals = np.linspace(domain[0], domain[1], 100)
    y_vals = spline(x_vals)
    max_y_abs = np.max(np.abs(y_vals))
    if max_y_abs > 1:
        control_points_sorted[:, 1] = control_points_sorted[:, 1] / max_y_abs
    return control_points_sorted


def fixed_brush():
    control_points = sorted([(-1.0, 0.0),(0.0, 1.), (1.0, 0.0)])
    # control_points = sorted([(-1, 0.0), (-0.6, -0.25), (0.0, 1), (0.6, -0.25), (1.0, 0.0)])
    # control_points = sorted([(-1, 0.0), (-0.6, -0.2), (-0.3, 0.5), (0.0, 1.0), (0.3, 0.5), (0.6, -0.2), (1.0, 0.0)])
    return np.array(control_points)

def main():   
    # brush_profile = create_brush_profile()
    brush_profile = fixed_brush()
    window = ENSDFWindow(resolution=(400, 400), brush_profile=brush_profile)
    pyglet.app.run()


if __name__ == '__main__':
    main()
