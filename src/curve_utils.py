"""
curve_utils.py  —  Core NURBS flight-path generation (no Blender UI).

Extracted from generate_random_curve.py so that generate_data.py and other
scripts can import the logic without registering the add-on panel/operator.
"""

import random
from dataclasses import dataclass, field

import bpy
import mathutils
from mathutils.bvhtree import BVHTree


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CurveConfig:
    num_points:              int   = 10
    start_height_fraction:   float = 0.4
    lift_fraction:           float = 0.04
    min_clearance_fraction:  float = 0.02
    max_collision_iters:     int   = 200
    curve_name:              str   = "FlightPath"


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

def get_scene_bounds(depsgraph):
    """World-space (min_co, max_co) across all mesh instances."""
    min_co = mathutils.Vector((float('inf'),)  * 3)
    max_co = mathutils.Vector((float('-inf'),) * 3)
    found  = False

    for inst in depsgraph.object_instances:
        if inst.object.type != 'MESH':
            continue
        found = True
        mat = inst.matrix_world
        for corner in inst.object.bound_box:
            world   = mat @ mathutils.Vector(corner)
            min_co.x = min(min_co.x, world.x)
            min_co.y = min(min_co.y, world.y)
            min_co.z = min(min_co.z, world.z)
            max_co.x = max(max_co.x, world.x)
            max_co.y = max(max_co.y, world.y)
            max_co.z = max(max_co.z, world.z)

    if not found:
        raise RuntimeError("No mesh instances found in scene.")
    return min_co, max_co


def build_scene_bvh(depsgraph):
    """Single world-space BVH tree from all mesh instances."""
    vertices = []
    polygons = []

    for inst in depsgraph.object_instances:
        obj = inst.object
        if obj.type != 'MESH':
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if not mesh or not mesh.polygons:
            eval_obj.to_mesh_clear()
            continue
        base = len(vertices)
        mat  = inst.matrix_world
        for v in mesh.vertices:
            vertices.append(mat @ v.co)
        for poly in mesh.polygons:
            polygons.append(tuple(base + i for i in poly.vertices))
        eval_obj.to_mesh_clear()

    if not polygons:
        raise RuntimeError("No mesh polygons found in any instance.")
    return BVHTree.FromPolygons(vertices, polygons)


# ---------------------------------------------------------------------------
# Collision detection
# ---------------------------------------------------------------------------

def point_is_clear(bvh, pt, min_clearance):
    """True if pt is farther than min_clearance from all surfaces and not inside a mesh."""
    loc, _, _, _ = bvh.find_nearest(pt, min_clearance)
    if loc is not None:
        return False
    hit_up   = bvh.ray_cast(pt, mathutils.Vector((0, 0,  1)))
    hit_down = bvh.ray_cast(pt, mathutils.Vector((0, 0, -1)))
    if hit_up[0] is not None and hit_down[0] is not None:
        return False
    return True


def segment_violates_clearance(bvh, p1, p2, min_clearance):
    length      = (p2 - p1).length
    num_samples = max(1, int(length / (min_clearance * 0.5))) + 1
    for i in range(num_samples + 1):
        if not point_is_clear(bvh, p1.lerp(p2, i / num_samples), min_clearance):
            return True
    return False


# ---------------------------------------------------------------------------
# Curve generation
# ---------------------------------------------------------------------------

def generate_random_points(min_co, max_co, num_points, start_z_frac):
    scene_height = max_co.z - min_co.z
    start_z = min_co.z + scene_height * start_z_frac
    return [
        mathutils.Vector((
            random.uniform(min_co.x, max_co.x),
            random.uniform(min_co.y, max_co.y),
            start_z,
        ))
        for _ in range(num_points)
    ]


def resolve_collisions(bvh, points, lift_step, min_clearance, max_iters):
    """Phase 1: lift colliding control-polygon segments until clear."""
    for iteration in range(max_iters):
        collision_found = False
        for i in range(len(points) - 1):
            if segment_violates_clearance(bvh, points[i], points[i + 1], min_clearance):
                collision_found = True
                points[i].z     += lift_step
                points[i + 1].z += lift_step
        if not collision_found:
            return points, iteration
    return points, max_iters


def _get_curve_world_points(curve_obj, depsgraph):
    """Sample world-space points on the evaluated NURBS curve."""
    mat      = curve_obj.matrix_world
    eval_obj = curve_obj.evaluated_get(depsgraph)

    try:
        mesh = bpy.data.meshes.new_from_object(eval_obj)
        if mesh and mesh.vertices:
            pts = [mat @ v.co.copy() for v in mesh.vertices]
            bpy.data.meshes.remove(mesh)
            return pts
        if mesh:
            bpy.data.meshes.remove(mesh)
    except Exception:
        pass

    try:
        mesh = eval_obj.to_mesh()
        if mesh and mesh.vertices:
            pts = [mat @ v.co.copy() for v in mesh.vertices]
            eval_obj.to_mesh_clear()
            return pts
        eval_obj.to_mesh_clear()
    except Exception:
        pass

    # Last resort: dense linear interpolation along the control polygon
    spline = curve_obj.data.splines[0]
    ctrl   = [mat @ mathutils.Vector(spline.points[i].co[:3])
              for i in range(len(spline.points))]
    pts    = []
    for i in range(len(ctrl) - 1):
        n = max(4, int((ctrl[i + 1] - ctrl[i]).length * 8))
        for j in range(n):
            pts.append(ctrl[i].lerp(ctrl[i + 1], j / n))
    pts.append(ctrl[-1])
    return pts


def resolve_evaluated_collisions(bvh, curve_obj, min_clearance, lift_step, max_iters):
    """Phase 2: lift control points wherever the smooth NURBS curve still violates clearance."""
    spline = curve_obj.data.splines[0]
    n      = len(spline.points)
    mat    = curve_obj.matrix_world

    for _ in range(max_iters):
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        pts       = _get_curve_world_points(curve_obj, depsgraph)
        if not pts:
            return

        violations = [p for p in pts if not point_is_clear(bvh, p, min_clearance)]
        if not violations:
            return

        ctrl    = [mat @ mathutils.Vector(spline.points[i].co[:3]) for i in range(n)]
        to_lift = set()
        for vp in violations:
            nearest = min(range(n), key=lambda i: (ctrl[i] - vp).length_squared)
            for idx in range(max(0, nearest - 1), min(n, nearest + 2)):
                to_lift.add(idx)

        for idx in to_lift:
            x, y, z, w = spline.points[idx].co
            spline.points[idx].co = (x, y, z + lift_step, w)


def create_nurbs_curve(points, name):
    """Create a NURBS path object in the active collection."""
    curve_data            = bpy.data.curves.new(name=name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 16

    spline = curve_data.splines.new('NURBS')
    spline.points.add(len(points) - 1)
    spline.use_endpoint_u = True

    for i, pt in enumerate(points):
        spline.points[i].co = (pt.x, pt.y, pt.z, 1.0)

    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)
    return obj


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def generate_flight_curve(config: CurveConfig):
    """
    Full pipeline: build BVH → scatter points → resolve collisions → create curve.

    Returns (curve_obj, min_co, max_co).
    """
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    min_co, max_co = get_scene_bounds(depsgraph)
    bvh            = build_scene_bvh(depsgraph)

    scene_height  = max_co.z - min_co.z
    lift_step     = scene_height * config.lift_fraction
    min_clearance = scene_height * config.min_clearance_fraction

    points = generate_random_points(
        min_co, max_co, config.num_points, config.start_height_fraction
    )
    points, _ = resolve_collisions(bvh, points, lift_step, min_clearance, config.max_collision_iters)

    curve_obj = create_nurbs_curve(points, config.curve_name)
    resolve_evaluated_collisions(bvh, curve_obj, min_clearance, lift_step, config.max_collision_iters)

    return curve_obj, min_co, max_co
