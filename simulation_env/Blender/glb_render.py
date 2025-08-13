import os
import json
import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)

# === CONFIG ===
GLB_PATH = "abandoned city.glb"
OUT_DIR = "glb_renders"
CAM_JSON = "camera_transform_data.json" 
IMAGE_SIZE = 512

# === Setup ===
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# === Load glb using trimesh ===
tm = trimesh.load(GLB_PATH, force="mesh")
verts = torch.tensor(tm.vertices, dtype=torch.float32, device=device).unsqueeze(0)
faces = torch.tensor(tm.faces, dtype=torch.int64, device=device).unsqueeze(0)

# Load vertex colors if available
if hasattr(tm.visual, 'vertex_colors') and tm.visual.vertex_colors is not None:
    colors = torch.tensor(tm.visual.vertex_colors[:, :3] / 255.0, dtype=torch.float32, device=device).unsqueeze(0)
    textures = TexturesVertex(verts_features=colors)
else:
    textures = None

mesh = Meshes(verts=verts, faces=faces, textures=textures)

# === Renderer Setup ===
raster_settings = RasterizationSettings(image_size=IMAGE_SIZE)
lights = PointLights(device=device, location=[[0.0, 1.0, 2.0]])

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, lights=lights)
)

# === Load camera poses (optional) ===
if CAM_JSON:
    with open(CAM_JSON, "r") as f:
        cam_data = json.load(f)
    frames = sorted(cam_data.items(), key=lambda x: int(x[0]))
else:
    frames = [(0, None)]

# === Render each frame ===
for idx, (frame_id, pose) in enumerate(frames):
    if pose is not None:
        R = torch.tensor(pose["R"], dtype=torch.float32).unsqueeze(0).to(device)
        T = torch.tensor(pose["T"], dtype=torch.float32).unsqueeze(0).to(device)

        # Optional: convert Blender → PyTorch3D coordinate system
        blender2p3d = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32).to(device)
        R = blender2p3d @ R
        T = blender2p3d @ T

        cameras = PerspectiveCameras(device=device, R=R, T=T)
    else:
        # Default camera (orbiting look-at)
        R, T = look_at_view_transform(dist=2.7, elev=20, azim=30 + 15 * idx)
        cameras = PerspectiveCameras(device=device, R=R, T=T)

    images = renderer(mesh, cameras=cameras)
    image = images[0, ..., :3].cpu().numpy()

    fname = os.path.join(OUT_DIR, f"frame_{int(frame_id):04d}.png")
    plt.imsave(fname, image)

    print(f"Saved {fname}")
