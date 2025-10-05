import bpy
import json

camera_name = "Camera"
output_path = bpy.path.abspath("//camera_transform_data.json")

camera = bpy.data.objects.get(camera_name)
if camera is None:
    raise Exception(f"Camera '{camera_name}' not found!")

transform_data = {}
scene = bpy.context.scene
start_frame = scene.frame_start
end_frame = scene.frame_end

for frame in range(start_frame, end_frame + 1):
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_camera = camera.evaluated_get(depsgraph)

    transform_data[frame] = {
        "location": [round(v, 6) for v in eval_camera.matrix_world.to_translation()],
        "rotation_euler": [round(v, 6) for v in eval_camera.matrix_world.to_euler()]
    }

with open(output_path, 'w') as f:
    json.dump(transform_data, f, indent=4)

print(f"Transform data saved to: {output_path}")
