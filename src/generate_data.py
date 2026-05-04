"""
generate_data.py  —  Blender batch drone-path data generator.

Usage:
    blender --background path/to/scene.blend --python src/generate_data.py -- \
        --num_paths 5 --output_dir /path/to/output [--optix]

Output layout:
    output_dir/
        path_0/
            ground_truth.txt      # one line per frame: frame x y z qx qy qz qw
            0001.png, 0002.png …  # rendered frames (format set by the .blend file)
        path_1/
        …
"""

import sys
import os
import argparse

import bpy
import mathutils

# Make sibling modules importable when Blender is invoked with --python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from curve_utils import CurveConfig, generate_flight_curve
from camera_utils import attach_camera_to_path, calc_gt


# ---------------------------------------------------------------------------
# Look-at strategies
#
# A look-at function receives (min_co, max_co) as mathutils.Vectors and
# returns the world-space mathutils.Vector the camera should point at.
# Swap `center_lookat` below for any other callable with that signature.
# ---------------------------------------------------------------------------

def center_lookat(min_co: mathutils.Vector, max_co: mathutils.Vector) -> mathutils.Vector:
    """Camera always looks at the horizontal centre of the scene."""
    return (min_co + max_co) / 2


# ---------------------------------------------------------------------------
# Single-path generation
# ---------------------------------------------------------------------------

def generate_path(index: int, output_dir: str, config: CurveConfig, lookat_fn=center_lookat):
    path_dir = os.path.join(output_dir, f"path_{index}")
    os.makedirs(path_dir, exist_ok=True)

    # 1. Generate curve
    config.curve_name = f"FlightPath_{index}"
    curve_obj, min_co, max_co = generate_flight_curve(config)

    # 2-3. Attach camera and set look-at
    lookat_point = lookat_fn(min_co, max_co)
    cam_obj, lookat_obj = attach_camera_to_path(bpy.context, curve_obj, lookat_point)

    # 4a. Render animation using settings already in the .blend file
    scene = bpy.context.scene
    prev_filepath        = scene.render.filepath
    scene.render.filepath = os.path.join(path_dir, "")
    bpy.ops.render.render(animation=True)
    scene.render.filepath = prev_filepath

    # 4b. Ground-truth camera poses
    calc_gt(os.path.join(path_dir, "ground_truth.txt"), camera_name=cam_obj.name)

    # 5. Delete the curve and look-at empty; keep the camera for the next path
    curve_data = curve_obj.data
    bpy.data.objects.remove(lookat_obj, do_unlink=True)
    bpy.data.objects.remove(curve_obj,  do_unlink=True)
    bpy.data.curves.remove(curve_data)

    print(f"[generate_data] path_{index} done → {path_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def enable_optix():
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for device in prefs.devices:
        device.use = True
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    print("[generate_data] OptiX GPU rendering enabled.")


def parse_args():
    argv = sys.argv
    argv = argv[argv.index('--') + 1:] if '--' in argv else []
    parser = argparse.ArgumentParser(description="Blender batch drone-path data generator")
    parser.add_argument('--num_paths',  type=int, required=True, help="Number of paths to generate")
    parser.add_argument('--output_dir', type=str, required=True, help="Root output directory")
    parser.add_argument('--optix', action='store_true', help="Enable OptiX GPU rendering (requires NVIDIA GPU + driver ≥ 520)")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.optix:
        enable_optix()

    config = CurveConfig()

    print(f"[generate_data] Generating {args.num_paths} paths → {args.output_dir}")
    for i in range(args.num_paths):
        print(f"[generate_data] --- path_{i} ---")
        generate_path(i, args.output_dir, config)

    print("[generate_data] Done.")


main()
