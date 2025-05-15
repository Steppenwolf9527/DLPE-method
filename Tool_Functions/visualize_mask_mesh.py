import numpy as np
import trimesh
from skimage import measure
import os

def mask_to_mesh(mask: np.ndarray) -> trimesh.Trimesh:
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh

def generate_combined_mask_mesh(
    lung_mask: np.ndarray,
    airway_mask: np.ndarray,
    vessel_mask: np.ndarray,
    output_path: str = "combined_lung_model.glb"
):
    masks = {
        "lung": lung_mask,
        "airway": airway_mask,
        "vessel": vessel_mask
    }

    colors = {
        "lung": [135, 206, 250, 150],      # light blue
        "airway": [0, 255, 0, 150],        # green
        "vessel": [255, 0, 0, 150],        # red
    }

    group_meshes = []

    for name, mask in masks.items():
        print(f"Generating mesh for {name}")
        mesh = mask_to_mesh(mask)
        mesh.visual.face_colors = colors[name]
        mesh.metadata["name"] = name
        group_meshes.append(mesh)

    scene = trimesh.Scene()
    for mesh in group_meshes:
        scene.add_geometry(mesh, node_name=mesh.metadata["name"])

    scene.export(output_path)
    print(f"Saved combined mask mesh to {output_path}")
