from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import numpy as np
import trimesh

from scipy.interpolate import CubicSpline
from .sampling import sample_uniform_disk, sample_along_spline
from .geoutils import project_on_surface, tangent_grad, intersect_plane, \
                      linear_fall, cubic_fall, quintic_fall, exp_fall, intrude_fall, wave_fall, ellipse
from .diff_operators import gradient

def intensity_modulation(t, mode="identity"):
    """
    Modulate brush intensity along the curve.
    t : float [0, 1] : Position along the curve (0 = start, 1 = end)
    mode : str : Type of modulation ('linear', 'exponential', 'sinusoidal')
    Returns a scaling factor for brush intensity.
    """
    if mode == "linear":
        return 1 - t  # Linearly fade out from 1 to 0
    elif mode == "exponential":
        return torch.exp(- 6 * t)  # Exponentially fade out
    elif mode == "central":
        return torch.exp(-((t - 0.5) ** 2) * 6) 
    elif mode == "inverse_central":
        return 1 - torch.exp(-((t - 0.5) ** 2) * 6)  # Low at center, increases toward ends
    elif mode == "sinusoidal":
        return 1.0 * (1 + torch.sin(10 * (t - 0.5))) 
    elif mode=="identity":
        return torch.ones_like(t)  # Default: no modulation


class BrushBase(ABC):
    def __init__(self, radius=0.07, intensity=0.04):
        self.radius = radius
        self.intensity = intensity
        self.inter_point = self.inter_normal = self.inter_sdf = None
    
    def set_interaction(self, inter_point, inter_normal, inter_sdf):
        self.inter_point  = inter_point
        self.inter_normal = F.normalize(inter_normal, dim=-1)
        self.inter_sdf    = inter_sdf

    def inside_interaction(self, points):
        return (points - self.inter_point).norm(dim=-1) <= self.radius

    @abstractmethod
    def sample_interaction(self, model, num_samples):
        raise NotImplemented


class SimpleBrush(BrushBase):
    def __init__(self, brush_type, **kwargs):
        super().__init__(**kwargs)
        self.brush_type = brush_type

    @property
    def brush_type(self):
        return self._brush_type

    @brush_type.setter
    def brush_type(self, brush_type):
        self._brush_type = brush_type

        if brush_type == 'linear':
            self.template = linear_fall
        elif brush_type == 'cubic':
            self.template = cubic_fall
        elif brush_type == 'quintic':
            self.template = quintic_fall
        elif brush_type == 'exp':
            self.template = exp_fall
        elif brush_type == 'wave':
            self.template = wave_fall
        elif brush_type == 'intrude':
            self.template = intrude_fall
        elif brush_type=='2D':
            self.template = ellipse
        else:
            raise ValueError(f'{brush_type} is not a valid type for {type(self).__name__}')
    
    def evaluate_template_on_tangent_disk(self, disk_points):
        if self.brush_type!="2D":
            disk_sample_norms = torch.norm(disk_points, dim=-1)
            disk_sample_norms.requires_grad = True
            y = self.intensity * self.template(disk_sample_norms / self.radius)
            dy = gradient(y, disk_sample_norms)
            y, dy = y.detach().unsqueeze(-1), dy.detach().unsqueeze(-1)
            return y, dy
        else:
            points = torch.empty_like(disk_points).copy_(disk_points)
            points.requires_grad = True
            z = self.intensity * self.template(points / self.radius)
            dz = torch.autograd.grad(outputs=z, inputs=points, 
                                 grad_outputs=torch.ones_like(z),
                                 create_graph=True)[0]
        
            z, dz = z.detach().unsqueeze(-1), dz.detach() 
            return z, dz
    

    def adjust_normals(self, projected_normals, disk_points, template_grad):
        normals = F.normalize(disk_points, dim=-1)
        normals.mul_(template_grad)
        tan_grad = tangent_grad(projected_normals, self.inter_normal)
        normals.add_(tan_grad)
        torch.add(self.inter_normal, normals, alpha=-1, out=normals)
        F.normalize(normals, dim=-1, out=normals)
        return normals

    def sample_interaction(self, model, num_samples):
        disk_samples = sample_uniform_disk(self.inter_normal, num_samples)
        disk_samples = disk_samples.squeeze(0) * self.radius
        y, dy = self.evaluate_template_on_tangent_disk(disk_samples)

        points = torch.add(disk_samples, self.inter_point)
        projected_points, projected_sdf, projected_normals = project_on_surface(
            model,
            points,
            num_steps=2
        )

        points = torch.addcmul(
            projected_points,
            y,
            self.inter_normal,
            out=projected_points
        )

        normals = self.adjust_normals(projected_normals, disk_samples, dy)

        sdf = torch.zeros(num_samples, 1, device=self.inter_point.device)

        return points, sdf, normals

    def deform_mesh(self, mesh):
        mesh_points = torch.tensor(
            mesh.vertices, dtype=torch.float32, device=self.inter_point.device
        )
        mesh_normals = torch.tensor(
            mesh.vertex_normals, dtype=torch.float32, device=self.inter_point.device
        )

        inside = self.inside_interaction(mesh_points)
        points_in_interaction = mesh_points[inside]
        normals_in_interaction = mesh_normals[inside]

        plane_points = intersect_plane(self.inter_normal, self.inter_point, points_in_interaction, normals_in_interaction)
        plane_points -= self.inter_point

        y, dy = self.evaluate_template_on_tangent_disk(plane_points)

        adjusted_normals = self.adjust_normals(normals_in_interaction, plane_points, dy)
        
        mesh_points[inside] = points_in_interaction + y * self.inter_normal
        mesh_normals[inside] = adjusted_normals

        deformed_mesh = trimesh.Trimesh(
            vertices=mesh_points.cpu(),
            vertex_normals=mesh_normals.cpu(),
            faces=mesh.faces
        )

        return deformed_mesh


class StrokeBrush(BrushBase):
    def __init__(self, brush_type, brush_profile=None, template=None, tubular=True, mode="identity", **kwargs):
        super().__init__(**kwargs)
        self.brush_type = brush_type
        self.brush_profile = brush_profile
        self.path = None
        self.template = template
        self.tubular = tubular
        self.mode = mode
        self.curve = None

    @property
    def brush_type(self):
        return self._brush_type
    
    def set_curve(self, spline):
        self.curve = spline
    
    def set_interaction(self, inter_point, inter_normal, inter_sdf):
        self.inter_point  = inter_point
        self.inter_normal = F.normalize(inter_normal, dim=-1)
        self.inter_sdf    = inter_sdf

    @brush_type.setter
    def brush_type(self, brush_type):
        self._brush_type = brush_type
    
    def evaluate_template_on_tangent_disk(self, disk_points):
        if self.brush_type!="2D":
            disk_sample_norms = torch.norm(disk_points, dim=-1)
            disk_sample_norms.requires_grad = True
            y = self.intensity * self.template(disk_sample_norms.cpu().detach().numpy() / self.radius)
            y = torch.tensor(y).to(self.inter_point.device)
            # dy = gradient(y, disk_sample_norms)
            dy = torch.clone(y)
            y, dy = y.detach().unsqueeze(-1), dy.detach().unsqueeze(-1)
            return y, dy
        else:
            points = torch.empty_like(disk_points).copy_(disk_points)
            points.requires_grad = True
            z = self.intensity * self.template(points / self.radius)
            dz = torch.autograd.grad(outputs=z, inputs=points, 
                                 grad_outputs=torch.ones_like(z),
                                 create_graph=True)[0]
        
            z, dz = z.detach().unsqueeze(-1), dz.detach() 
            return z, dz
    
    def inside_interaction(self, points):
        """
        Determines if the points are inside the interaction region (within the tube along the spline).
        Args:
            points (torch.Tensor): A tensor of points to test, of shape (n, 3) or (n, 2).
        Returns:
            torch.Tensor: A boolean tensor of shape (n,) indicating whether each point is inside the interaction region.
        """
        distances = torch.cdist(points.unsqueeze(0), self.inter_point.unsqueeze(0)).squeeze(0)
        min_distances = torch.min(distances, dim=1)[0]
        return min_distances <= self.radius

    
    def evaluate_template_on_tube(self, radii, modulation_factors):
        modulation_factors = modulation_factors.reshape(-1, 1).to(self.inter_point.device)
        normalized_distances = radii / self.radius
        batch_cpu = normalized_distances.cpu().numpy()
        y_cpu = self.template(batch_cpu)
        y = torch.tensor(y_cpu, device=self.inter_point.device, dtype=radii.dtype)
        template_derivative = self.template.derivative()
        dy_cpu = template_derivative(batch_cpu)
        dy = torch.tensor(dy_cpu, device=self.inter_point.device, dtype=radii.dtype)
        y = (self.intensity * modulation_factors * y).unsqueeze(-1).expand(-1, -1, 3)
        dy = (self.intensity * modulation_factors * dy).unsqueeze(-1).expand(-1, -1, 3)
        return y, dy

    def adjust_normals(self, projected_normals, disk_points, template_grad):
        normals = F.normalize(disk_points, dim=-1)
        normals.mul_(template_grad)
        tan_grad = tangent_grad(projected_normals, self.inter_normal.unsqueeze(1))
        normals.add_(tan_grad)
        torch.add(self.inter_normal.unsqueeze(1), normals, alpha=-1, out=normals)
        F.normalize(normals, dim=-1, out=normals)
        return normals

    def sample_interaction(self, model, num_samples):
        if not self.tubular or not self.curve:
            disk_samples = sample_uniform_disk(self.inter_normal, num_samples)
            disk_samples = disk_samples.squeeze(0) * self.radius
            if not self.curve:
                disk_samples = disk_samples.unsqueeze(0)
            y, dy = self.evaluate_template_on_tangent_disk(disk_samples)
            points = torch.add(disk_samples, self.inter_point.unsqueeze(1))
            projected_points, projected_sdf, projected_normals = project_on_surface(
                model,
                points,
                num_steps=2
            )
            points = torch.addcmul(
                projected_points,
                y,
                self.inter_normal.unsqueeze(1),
                out=projected_points
            )
            normals = self.adjust_normals(projected_normals, disk_samples, dy)

            sdf = torch.zeros(normals.shape[0]*normals.shape[1], 1, device=self.inter_point.device)
            return points, sdf, normals
        else:
            tube_samples, radii = sample_along_spline(
                self.curve, self.inter_point, self.inter_normal,
                radius=self.radius, num_offsets=101
            )
            modulation_factors = intensity_modulation(
                torch.linspace(0, 1, self.inter_point.shape[0]),
                mode=self.mode
            )
            y, dy = self.evaluate_template_on_tube(radii, modulation_factors)
            N, M = tube_samples.shape[:2]
            inter_point = self.inter_point.unsqueeze(1)
            inter_normal = self.inter_normal.unsqueeze(1)
            points = tube_samples + inter_point
            projected_points, projected_sdf, projected_normals = project_on_surface(
                model, points, num_steps=2
            )
            points = torch.addcmul(
                projected_points.reshape(N * M, 3),
                y.reshape(N * M, 3),
                inter_normal.expand(N, M, 3).reshape(N * M, 3)
            ).reshape(N, M, 3)

            normals = self.adjust_normals(projected_normals, tube_samples, dy)
            sdf = normals.new_zeros(normals.shape[0] * normals.shape[1], 1)
            return points, sdf, normals



    def deform_mesh(self, mesh):
        mesh_points = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.inter_point.device)
        mesh_normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32, device=self.inter_point.device)
        deformation_values = torch.zeros_like(mesh_points)
        min_distance_to_segment = torch.full((mesh_points.shape[0],), float('inf'), device=self.inter_point.device)
        modulation_factors = intensity_modulation(torch.linspace(0, 1, self.inter_point.shape[0]), mode=self.mode)
        vertex_colors = np.full((mesh_points.shape[0], 4), [255, 255, 255, 255], dtype=np.uint8)

        for i in range(self.inter_point.shape[0] - 1):
            start_point = self.inter_point[i]
            end_point = self.inter_point[i + 1]
            segment_vector = end_point - start_point
            start_point_idx = torch.argmin(torch.norm(mesh_points - start_point, dim=1))
            end_point_idx = torch.argmin(torch.norm(mesh_points - end_point, dim=1))
            vertex_colors[start_point_idx] = [255, 0, 0, 255]
            vertex_colors[end_point_idx] = [255, 0, 0, 255]

            start_to_mesh = mesh_points - start_point
            projection_lengths_current = (start_to_mesh * segment_vector).sum(dim=1) / segment_vector.norm() ** 2
            projection_points_current = start_point + projection_lengths_current.unsqueeze(1) * segment_vector

            projection_points_current = torch.where(
                projection_lengths_current.unsqueeze(1) < 0,
                start_point,
                torch.where(projection_lengths_current.unsqueeze(1) > 1, end_point, projection_points_current)
            )

            distance_to_segment = torch.norm(mesh_points - projection_points_current, dim=1)

            valid_indices = torch.nonzero(distance_to_segment <= self.radius).squeeze(1)
            normalized_distances = distance_to_segment[valid_indices] / self.radius
            deformation_intensity = (
                torch.tensor(self.template(normalized_distances.detach().cpu().numpy()), dtype=torch.float32, device=self.inter_point.device)
                * self.intensity * modulation_factors[i]
            )
            closer_vertices = distance_to_segment[valid_indices] < min_distance_to_segment[valid_indices]
            valid_indices_closer = valid_indices[closer_vertices]
            deformation_values[valid_indices_closer] = deformation_intensity[closer_vertices].unsqueeze(1) * mesh_normals[valid_indices_closer]
            min_distance_to_segment[valid_indices_closer] = distance_to_segment[valid_indices_closer]
            vertex_colors[valid_indices_closer.cpu().numpy()] = [0, 0, 255, 255]  

        deformed_mesh_points = mesh_points + deformation_values
        deformed_mesh = trimesh.Trimesh(
            vertices=deformed_mesh_points.cpu().numpy(),
            vertex_normals=mesh_normals.cpu().numpy(),
            faces=mesh.faces,
            # vertex_colors=vertex_colors
        )
        return deformed_mesh