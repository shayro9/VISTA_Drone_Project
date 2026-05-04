"""
camera_utils.py  —  Camera attachment and ground-truth export for Blender.
"""

import bpy
import mathutils


def attach_camera_to_path(context, curve_obj, lookat_point, animate=True):
    """
    Attach the scene camera to curve_obj with Follow Path + Track To constraints.

    lookat_point: mathutils.Vector — world-space point the camera looks at.
    Returns (cam_obj, lookat_empty).
    """
    scene      = context.scene
    collection = context.collection or scene.collection

    cam_obj = scene.camera
    if cam_obj is None:
        cam_data = bpy.data.cameras.new("FlightCamera")
        cam_obj  = bpy.data.objects.new("FlightCamera", cam_data)
        collection.objects.link(cam_obj)
        scene.camera = cam_obj

    # Remove stale constraints from a previous path
    for c in list(cam_obj.constraints):
        if c.type in {'FOLLOW_PATH', 'TRACK_TO'}:
            cam_obj.constraints.remove(c)

    # Follow Path
    fp             = cam_obj.constraints.new('FOLLOW_PATH')
    fp.name        = "FollowFlightPath"
    fp.target      = curve_obj
    fp.use_fixed_location = True
    fp.offset_factor      = 0.0
    fp.use_curve_follow   = False

    # Look-at empty
    target_name = f"{curve_obj.name}_LookAt"
    lookat_obj  = bpy.data.objects.new(target_name, None)
    lookat_obj.empty_display_type = 'SPHERE'
    lookat_obj.empty_display_size = 0.5
    lookat_obj.location           = lookat_point
    collection.objects.link(lookat_obj)

    # Track To
    tt            = cam_obj.constraints.new('TRACK_TO')
    tt.target     = lookat_obj
    tt.track_axis = 'TRACK_NEGATIVE_Z'
    tt.up_axis    = 'UP_Y'

    if animate:
        prev_active   = context.view_layer.objects.active
        prev_selected = list(context.selected_objects)

        bpy.ops.object.select_all(action='DESELECT')
        cam_obj.select_set(True)
        context.view_layer.objects.active = cam_obj
        bpy.ops.constraint.followpath_path_animate(constraint=fp.name, owner='OBJECT')

        bpy.ops.object.select_all(action='DESELECT')
        for o in prev_selected:
            o.select_set(True)
        context.view_layer.objects.active = prev_active

    return cam_obj, lookat_obj


def calc_gt(output_path, camera_name="Camera"):
    """
    Write per-frame camera world-space pose to output_path.

    Format (one line per frame):
        frame  x  y  z  qx  qy  qz  qw
    """
    camera = bpy.data.objects.get(camera_name)
    if camera is None:
        raise RuntimeError(f"Camera '{camera_name}' not found.")

    scene = bpy.context.scene
    with open(output_path, 'w') as f:
        for frame in range(scene.frame_start, scene.frame_end + 1):
            scene.frame_set(frame)
            depsgraph   = bpy.context.evaluated_depsgraph_get()
            eval_camera = camera.evaluated_get(depsgraph)
            loc = eval_camera.matrix_world.to_translation()
            rot = eval_camera.matrix_world.to_quaternion()
            f.write(
                f"{frame} "
                f"{loc.x:.6f} {loc.y:.6f} {loc.z:.6f} "
                f"{rot.x:.6f} {rot.y:.6f} {rot.z:.6f} {rot.w:.6f}\n"
            )
