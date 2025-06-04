import torch
import json
import os
from tqdm import tqdm
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    Textures
)
from matplotlib import pyplot as plt

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the mesh
obj_path = "simulation_env/Blender/abandoned city.obj"
mesh = load_objs_as_meshes([obj_path], device=device)

# Load JSON camera poses
with open("simulation_env/Blender/camera_transform_data.json", "r") as f:
    cam_data = json.load(f)

# Renderer setup
raster_settings = RasterizationSettings(image_size=512)
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, lights=lights)
)

# Render per frame
out_dir = "renders"
os.makedirs(out_dir, exist_ok=True)

for frame, pose in tqdm(sorted(cam_data.items(), key=lambda x: int(x[0]))):
    R = torch.tensor(pose["R"], dtype=torch.float32).unsqueeze(0).to(device)
    T = torch.tensor(pose["T"], dtype=torch.float32).unsqueeze(0).to(device)

    camera = PerspectiveCameras(device=device, R=R, T=T)

    images = renderer(mesh, cameras=camera)
    image = images[0, ..., :3].cpu().numpy()

    plt.imsave(f"{out_dir}/frame_{int(frame):04d}.png", image)
