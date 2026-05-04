import bpy

def calc_gt(output_path):
	camera_name = "Camera"

	camera = bpy.data.objects.get(camera_name)
	if camera is None:
	    raise Exception(f"Camera '{camera_name}' not found!")

	scene = bpy.context.scene
	start_frame = scene.frame_start
	end_frame = scene.frame_end

	with open(output_path, 'w') as f:
	    for frame in range(start_frame, end_frame + 1):
	        scene.frame_set(frame)
        	depsgraph = bpy.context.evaluated_depsgraph_get()
	        eval_camera = camera.evaluated_get(depsgraph)

        	location = eval_camera.matrix_world.to_translation()
	        rotation = eval_camera.matrix_world.to_quaternion()

        	line = f"{frame} {location.x:.6f} {location.y:.6f} {location.z:.6f} {rotation.x:.	6f} {rotation.y:.6f} {rotation.z:.6f} {rotation.w:.6f}\n"
	        f.write(line)

	print(f"Transform data saved to: {output_path}")

