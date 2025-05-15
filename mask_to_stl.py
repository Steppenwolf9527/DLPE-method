import numpy as np
import trimesh
from skimage import measure
import os

def mask_to_mesh(mask: np.ndarray) -> trimesh.Trimesh:
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh

def load_mask(path):
    data = np.load(path)
    print(f"Loading from {path}, keys: {list(data.keys())}")
    return data[list(data.keys())[0]]
def main():
    masks = {
        "lung": "C:/Users/烟雨平生/Downloads/lung_mask.npz",
        "airway": "C:/Users/烟雨平生/Downloads/airways_mask.npz",
        "vessel": "C:/Users/烟雨平生/Downloads/blood_vessel_mask.npz"
    }

    group_meshes = []
    colors = {
        "lung": [135, 206, 250, 150],      # light blue
        "airway": [0, 255, 0, 150],        # green
        "vessel": [255, 0, 0, 150],        # red
    }

    for name, path in masks.items():
        print(f"Processing {name}")
        mask = load_mask(path)
        mesh = mask_to_mesh(mask)
        mesh.visual.face_colors = colors[name]
        mesh.metadata["name"] = name
        group_meshes.append(mesh)

    scene = trimesh.Scene()
    for mesh in group_meshes:
        scene.add_geometry(mesh, node_name=mesh.metadata["name"])

    scene.export("combined_lung_model.glb")
    print("Saved as combined_lung_model.glb")

if __name__ == "__main__":
    main()
