# optimize_light_with_scene.py
import torch
import slangpy
import trimesh
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# === Load camera poses ===
def pose_to_matrix(location, rotation_euler):
    rot = R.from_euler('XYZ', rotation_euler).as_matrix()
    rot = torch.tensor(rot, dtype=torch.float32)
    trans = torch.tensor(location, dtype=torch.float32)
    cam_to_world = torch.eye(4)
    cam_to_world[:3, :3] = rot
    cam_to_world[:3, 3] = trans
    return torch.inverse(cam_to_world)

with open('camera_transform_data.json') as f:
    poses_json = json.load(f)

camera_poses = []
for frame_id in sorted(poses_json.keys(), key=int):
    pose = poses_json[frame_id]
    location = pose["location"]
    rotation = pose["rotation_euler"]
    cam_matrix = pose_to_matrix(location, rotation)
    camera_poses.append(cam_matrix)

# === Load mesh from OBJ ===
mesh = trimesh.load('textures/abandoned_city.obj')
vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
triangles = torch.tensor(mesh.faces, dtype=torch.int32)

# === Load Slang shader ===
module = slangpy.Module.from_file("light_renderer.slang")
render_frame = module.renderScene

# === Define UV grid ===
H, W = 64, 64
uvs = torch.stack(torch.meshgrid(
    torch.linspace(0, 1, H),
    torch.linspace(0, 1, W),
    indexing='ij'
), dim=-1).reshape(-1, 2)

# === Target image (can be replaced) ===
target_image = torch.zeros((H, W))
target_image[H//2, W//2] = 1.0

# === Learnable shared light source ===
light_pos = torch.tensor([3.0, 5.0, 2.0], requires_grad=True)
light_intensity = torch.tensor([1.0], requires_grad=True)

# === Optimization ===
optimizer = torch.optim.Adam([light_pos, light_intensity], lr=0.05)

for epoch in range(30):
    optimizer.zero_grad()
    total_loss = 0.0

    for cam_pose in camera_poses:
        pixel_vals = []
        for uv in uvs:
            brightness = render_frame(
                uv, cam_pose, vertices, triangles, light_pos, light_intensity
            )
            pixel_vals.append(brightness)
        image = torch.stack(pixel_vals).view(H, W)
        total_loss += ((image - target_image) ** 2).mean()

    total_loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")

# === Final output ===
with torch.no_grad():
    output_vals = []
    for uv in uvs:
        brightness = render_frame(
            uv, camera_poses[0], vertices, triangles, light_pos, light_intensity
        )
        output_vals.append(brightness)
    final_img = torch.stack(output_vals).view(H, W)

plt.imshow(final_img.numpy(), cmap='gray')
plt.title("Final Rendered Image")
plt.show()
