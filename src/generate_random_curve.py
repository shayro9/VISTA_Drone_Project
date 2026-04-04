"""
generate_random_curve.py  –  Blender Add-on

Generates a random 3D NURBS curve inside a Blender scene, scaled to the scene's
mesh geometry. Uses iterative collision resolution: when a segment between two
control points intersects existing meshes, both neighboring points are lifted in Z
and the check repeats until the curve is fully clear.

Install:  Edit → Preferences → Add-ons → Install → select this file → Enable
Panel:    3D Viewport → Sidebar (N) → Tool tab → "Flight Curve Generator"
"""

bl_info = {
    "name":        "Random Flight Curve Generator",
    "author":      "VISTA Drone Project",
    "version":     (1, 0, 0),
    "blender":     (3, 0, 0),
    "location":    "3D Viewport › Sidebar › Curve Gen",
    "description": "Generate a random collision-free NURBS curve scaled to the scene",
    "category":    "Curve",
}

import bpy
import random
import mathutils
from bpy.props import (IntProperty, FloatProperty, StringProperty,
                       BoolProperty, EnumProperty, FloatVectorProperty,
                       PointerProperty)
from bpy.types import Operator, Panel, PropertyGroup
from mathutils.bvhtree import BVHTree

# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class CurveGenProperties(PropertyGroup):
    num_points: IntProperty(
        name="Control Points",
        description="Number of control points on the generated curve",
        default=10, min=2, max=100,
    )
    start_height_fraction: FloatProperty(
        name="Start Height",
        description="Starting Z as a fraction of the full scene height (0 = floor, 1 = top)",
        default=0.4, min=0.0, max=1.0, subtype='FACTOR',
    )
    lift_fraction: FloatProperty(
        name="Lift Step",
        description="How much to lift colliding points per iteration, as a fraction of scene height",
        default=0.04, min=0.001, max=0.5, subtype='FACTOR',
    )
    max_collision_iters: IntProperty(
        name="Max Iterations",
        description="Safety cap on the collision-resolution loop",
        default=200, min=1, max=2000,
    )
    min_clearance_fraction: FloatProperty(
        name="Min Clearance",
        description="Minimum distance to keep from any surface, as a fraction of scene height. "
                    "Also controls how densely each segment is sampled",
        default=0.02, min=0.001, max=0.2, subtype='FACTOR',
    )
    curve_name: StringProperty(
        name="Curve Name",
        description="Name of the generated curve object",
        default="RandomFlightPath",
    )

    # --- Camera ---
    attach_camera: BoolProperty(
        name="Attach Camera to Path",
        description="Add a camera that moves along the generated curve",
        default=False,
    )
    use_existing_camera: BoolProperty(
        name="Use Scene Camera",
        description="Move the existing scene camera; creates a new one if none exists",
        default=True,
    )
    look_at_mode: EnumProperty(
        name="Look At",
        description="What the camera points at while travelling the path",
        items=[
            ('NONE',   "Path Direction", "Camera aligns itself to the curve tangent"),
            ('CENTER', "Scene Center",   "Camera always looks at the centre of the scene"),
            ('POINT',  "Fixed Point",    "Camera always looks at a specific world location"),
            ('OBJECT', "Object",         "Camera tracks a chosen object"),
        ],
        default='CENTER',
    )
    look_at_point: FloatVectorProperty(
        name="Look-At Point",
        description="World-space location the camera will look at",
        default=(0.0, 0.0, 0.0),
        subtype='XYZ',
    )
    look_at_object: PointerProperty(
        name="Look-At Object",
        type=bpy.types.Object,
        description="Object the camera will track",
    )
    animate_path: BoolProperty(
        name="Animate Along Path",
        description="Keyframe the camera travelling the full curve over the scene frame range",
        default=True,
    )


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

def get_scene_bounds(depsgraph):
    """
    Return (min_co, max_co) world-space bounding box of ALL mesh instances.

    Uses depsgraph.object_instances instead of bpy.context.scene.objects so
    that collection instances, Geometry Nodes instances, and dupli-objects
    (common in procedural city generators) are all included.
    """
    min_co = mathutils.Vector((float('inf'),) * 3)
    max_co = mathutils.Vector((float('-inf'),) * 3)
    found = False

    for inst in depsgraph.object_instances:
        if inst.object.type != 'MESH':
            continue
        found = True
        mat = inst.matrix_world
        for corner in inst.object.bound_box:
            world = mat @ mathutils.Vector(corner)
            min_co.x = min(min_co.x, world.x)
            min_co.y = min(min_co.y, world.y)
            min_co.z = min(min_co.z, world.z)
            max_co.x = max(max_co.x, world.x)
            max_co.y = max(max_co.y, world.y)
            max_co.z = max(max_co.z, world.z)

    if not found:
        raise RuntimeError(
            "No mesh instances found. Your city generator may use collection "
            "instances or Geometry Nodes — these are now supported, but make "
            "sure the viewport is not hidden and modifiers are applied."
        )
    return min_co, max_co


def build_scene_bvh(depsgraph):
    """
    Build a single world-space BVH tree from ALL mesh instances.

    Uses depsgraph.object_instances so collection instances, Geometry Nodes
    instances, and dupli-objects are included — critical for procedural city
    generators that don't create plain MESH scene objects.
    """
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
        mat = inst.matrix_world  # per-instance transform, not obj.matrix_world
        for v in mesh.vertices:
            vertices.append(mat @ v.co)
        for poly in mesh.polygons:
            polygons.append(tuple(base + i for i in poly.vertices))

        eval_obj.to_mesh_clear()

    if not polygons:
        raise RuntimeError(
            "No mesh polygons found in any instance. Check that your city "
            "generator has actually generated geometry and is visible."
        )
    return BVHTree.FromPolygons(vertices, polygons)


# ---------------------------------------------------------------------------
# Collision detection
# ---------------------------------------------------------------------------

def point_is_clear(bvh, pt, min_clearance):
    """
    Return True only if pt is:
      - farther than min_clearance from every surface (proximity check), AND
      - not enclosed inside a closed mesh (inside-mesh check).

    The inside-mesh check casts rays in opposite Z directions. BVHTree never
    culls back-faces, so a point inside a closed mesh will register hits in
    BOTH directions regardless of face normals.
    """
    # 1. Proximity: any surface closer than min_clearance?
    loc, _, _, _ = bvh.find_nearest(pt, min_clearance)
    if loc is not None:
        return False

    # 2. Containment: enclosed by geometry above AND below?
    hit_up   = bvh.ray_cast(pt, mathutils.Vector((0, 0,  1)))
    hit_down = bvh.ray_cast(pt, mathutils.Vector((0, 0, -1)))
    if hit_up[0] is not None and hit_down[0] is not None:
        return False

    return True


def segment_violates_clearance(bvh, p1, p2, min_clearance):
    """
    Sample densely along p1→p2 (spacing = min_clearance/2) and return True
    if any sample fails point_is_clear.
    """
    length = (p2 - p1).length
    num_samples = max(1, int(length / (min_clearance * 0.5))) + 1

    for i in range(num_samples + 1):
        if not point_is_clear(bvh, p1.lerp(p2, i / num_samples), min_clearance):
            return True
    return False


# ---------------------------------------------------------------------------
# Curve generation
# ---------------------------------------------------------------------------

def generate_random_points(min_co, max_co, num_points, start_z_frac):
    """
    Scatter control points randomly in XY within the scene footprint.
    Z is initialised at `start_z_frac` of the full scene height above the floor.
    """
    scene_height = max_co.z - min_co.z
    start_z = min_co.z + scene_height * start_z_frac

    points = []
    for _ in range(num_points):
        x = random.uniform(min_co.x, max_co.x)
        y = random.uniform(min_co.y, max_co.y)
        points.append(mathutils.Vector((x, y, start_z)))
    return points


def resolve_collisions(bvh, points, lift_step, min_clearance, max_iters):
    """
    Iteratively lift pairs of neighboring control points until no segment
    of the curve violates the minimum clearance from scene geometry.

    Returns (resolved_points, iterations_used).
    """
    for iteration in range(max_iters):
        collision_found = False

        for i in range(len(points) - 1):
            if segment_violates_clearance(bvh, points[i], points[i + 1], min_clearance):
                collision_found = True
                points[i].z += lift_step
                points[i + 1].z += lift_step

        if not collision_found:
            return points, iteration

    print(f"[generate_random_curve] Warning: reached max iterations ({max_iters}). "
          "Some collisions may remain.")
    return points, max_iters

def get_curve_world_points(curve_obj, depsgraph):
    """
    Return world-space points along the evaluated NURBS curve.

    Tries three methods in order of reliability:
      1. bpy.data.meshes.new_from_object() — most reliable across Blender versions
      2. eval_obj.to_mesh()               — older fallback
      3. Dense control-polygon sampling   — last resort if both tessellation APIs
                                            return nothing (common for unfilled 3D
                                            curves in some Blender builds)

    Returns (points, tessellated) where tessellated=False means we fell back to
    the control-polygon proxy.
    """
    mat = curve_obj.matrix_world
    eval_obj = curve_obj.evaluated_get(depsgraph)

    # Method 1: new_from_object (Blender 3.x+, works for curves without fill)
    try:
        mesh = bpy.data.meshes.new_from_object(eval_obj)
        if mesh and mesh.vertices:
            pts = [mat @ v.co.copy() for v in mesh.vertices]
            bpy.data.meshes.remove(mesh)
            return pts, True
        if mesh:
            bpy.data.meshes.remove(mesh)
    except Exception:
        pass

    # Method 2: to_mesh()
    try:
        mesh = eval_obj.to_mesh()
        if mesh and mesh.vertices:
            pts = [mat @ v.co.copy() for v in mesh.vertices]
            eval_obj.to_mesh_clear()
            return pts, True
        eval_obj.to_mesh_clear()
    except Exception:
        pass

    # Method 3: dense control-polygon fallback
    # Sample every min segment at 32 points — tighter than the NURBS deviation
    spline = curve_obj.data.splines[0]
    ctrl = [mat @ mathutils.Vector(spline.points[i].co[:3])
            for i in range(len(spline.points))]
    pts = []
    for i in range(len(ctrl) - 1):
        seg_len = (ctrl[i + 1] - ctrl[i]).length
        n = max(4, int(seg_len * 8))
        for j in range(n):
            pts.append(ctrl[i].lerp(ctrl[i + 1], j / n))
    pts.append(ctrl[-1])
    return pts, False  # False = tessellation failed, used proxy


def resolve_evaluated_collisions(bvh, curve_obj, min_clearance, lift_step, max_iters):
    """
    Phase 2: check the *actual rendered NURBS geometry* (not just the control
    polygon) and lift the nearest control points wherever the smooth curve
    violates clearance.

    The NURBS spline can deviate significantly from the straight control-polygon
    segments — this pass catches those dips that phase 1 misses.
    """
    spline = curve_obj.data.splines[0]
    n = len(spline.points)
    mat = curve_obj.matrix_world

    tessellation_ok = None  # will be set on first iteration

    for iteration in range(max_iters):
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()

        pts, tessellated = get_curve_world_points(curve_obj, depsgraph)
        if tessellation_ok is None:
            tessellation_ok = tessellated

        if not pts:
            # Nothing to evaluate — bail out and warn
            return -1

        violations = [p for p in pts if not point_is_clear(bvh, p, min_clearance)]

        if not violations:
            return iteration

        # Lift the control point nearest to each violation and its neighbours
        ctrl = [mat @ mathutils.Vector(spline.points[i].co[:3]) for i in range(n)]
        to_lift = set()
        for vp in violations:
            nearest = min(range(n), key=lambda i: (ctrl[i] - vp).length_squared)
            for idx in range(max(0, nearest - 1), min(n, nearest + 2)):
                to_lift.add(idx)

        for idx in to_lift:
            x, y, z, w = spline.points[idx].co
            spline.points[idx].co = (x, y, z + lift_step, w)

    return max_iters




def create_nurbs_curve(points, name):
    """Create a smooth NURBS path through the given world-space points."""
    curve_data = bpy.data.curves.new(name=name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 16

    spline = curve_data.splines.new('NURBS')
    spline.points.add(len(points) - 1)  # spline starts with 1 point
    spline.use_endpoint_u = True        # curve passes through first/last points

    for i, pt in enumerate(points):
        spline.points[i].co = (pt.x, pt.y, pt.z, 1.0)  # NURBS homogeneous coords

    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)
    return obj


# ---------------------------------------------------------------------------
# Camera setup
# ---------------------------------------------------------------------------

def attach_camera_to_curve(context, curve_obj, props, min_co, max_co):
    """
    Get or create a camera, attach it to curve_obj with a Follow Path
    constraint, and optionally add a Track To constraint so it looks at
    a target while travelling.
    """
    scene = context.scene
    collection = context.collection or scene.collection

    # ── Camera object ──────────────────────────────────────────────────────
    if props.use_existing_camera and scene.camera:
        cam_obj = scene.camera
    else:
        cam_data = bpy.data.cameras.new("FlightCamera")
        cam_obj = bpy.data.objects.new("FlightCamera", cam_data)
        collection.objects.link(cam_obj)
        scene.camera = cam_obj

    # Remove stale constraints from previous runs
    for c in list(cam_obj.constraints):
        if c.type in {'FOLLOW_PATH', 'TRACK_TO'}:
            cam_obj.constraints.remove(c)

    # ── Follow Path ────────────────────────────────────────────────────────
    fp = cam_obj.constraints.new('FOLLOW_PATH')
    fp.name = "FollowFlightPath"
    fp.target = curve_obj
    fp.use_fixed_location = True  # lets us drive position via offset_factor (0→1)
    fp.offset_factor = 0.0

    if props.look_at_mode == 'NONE':
        fp.use_curve_follow = True
        fp.forward_axis = 'TRACK_NEGATIVE_Z'
        fp.up_axis = 'UP_Y'
    else:
        fp.use_curve_follow = False

    # ── Track To ───────────────────────────────────────────────────────────
    if props.look_at_mode != 'NONE':
        if props.look_at_mode == 'OBJECT' and props.look_at_object:
            target_obj = props.look_at_object
        else:
            target_name = f"{curve_obj.name}_LookAt"
            target_obj = bpy.data.objects.get(target_name)
            if target_obj is None:
                target_obj = bpy.data.objects.new(target_name, None)
                target_obj.empty_display_type = 'SPHERE'
                target_obj.empty_display_size = 0.5
                collection.objects.link(target_obj)

            if props.look_at_mode == 'CENTER':
                target_obj.location = (min_co + max_co) / 2
            else:  # POINT
                target_obj.location = mathutils.Vector(props.look_at_point)

        tt = cam_obj.constraints.new('TRACK_TO')
        tt.target = target_obj
        tt.track_axis = 'TRACK_NEGATIVE_Z'
        tt.up_axis    = 'UP_Y'

    # ── Animation ──────────────────────────────────────────────────────────
    if props.animate_path:
        # Make the camera active — the operator requires it
        prev_active = context.view_layer.objects.active
        prev_selected = [o for o in context.selected_objects]
        bpy.ops.object.select_all(action='DESELECT')
        cam_obj.select_set(True)
        context.view_layer.objects.active = cam_obj

        # This is exactly what the "Animate Path" button calls internally.
        # It sets use_path, path_duration, and wires up eval_time keyframes.
        bpy.ops.constraint.followpath_path_animate(
            constraint=fp.name,
            owner='OBJECT'
        )

        # Restore previous selection
        bpy.ops.object.select_all(action='DESELECT')
        for o in prev_selected:
            o.select_set(True)
        context.view_layer.objects.active = prev_active
    else:
        fp.use_fixed_location = True
        fp.offset_factor = 0.0

    return cam_obj



class CURVEGEN_OT_generate(Operator):
    bl_idname = "curvegen.generate"
    bl_label = "Generate Curve"
    bl_description = "Generate a random collision-free NURBS curve scaled to the scene"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.curve_gen_props

        try:
            depsgraph = context.evaluated_depsgraph_get()
            min_co, max_co = get_scene_bounds(depsgraph)
            bvh = build_scene_bvh(depsgraph)
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        scene_height = max_co.z - min_co.z
        lift_step = scene_height * props.lift_fraction
        min_clearance = scene_height * props.min_clearance_fraction

        self.report({'INFO'}, f"Scene bounds: {min_co} → {max_co}  |  "
                              f"clearance: {min_clearance:.3f}  |  lift: {lift_step:.3f}")

        points = generate_random_points(
            min_co, max_co, props.num_points, props.start_height_fraction
        )
        points, iters1 = resolve_collisions(
            bvh, points, lift_step, min_clearance, props.max_collision_iters
        )

        curve_obj = create_nurbs_curve(points, props.curve_name)

        # Phase 2: check the actual smooth NURBS geometry, not just the control polygon
        iters2 = resolve_evaluated_collisions(
            bvh, curve_obj, min_clearance, lift_step, props.max_collision_iters
        )
        total_iters = iters1 + max(iters2, 0)

        if iters2 == -1:
            self.report({'WARNING'},
                        f"Created '{curve_obj.name}' but curve tessellation failed – "
                        "collision check used control-polygon proxy only.")
        elif iters1 >= props.max_collision_iters or iters2 >= props.max_collision_iters:
            self.report({'WARNING'},
                        f"Created '{curve_obj.name}' but hit max iterations "
                        f"(polygon: {iters1}, curve: {iters2}) – some collisions may remain.")
        else:
            self.report({'INFO'},
                        f"Created '{curve_obj.name}' ({props.num_points} pts) – "
                        f"resolved in {total_iters} iterations "
                        f"(polygon pass: {iters1}, curve pass: {iters2}).")

        # ── Optional camera ───────────────────────────────────────────────
        if props.attach_camera:
            try:
                cam = attach_camera_to_curve(context, curve_obj, props, min_co, max_co)
                self.report({'INFO'}, f"Camera '{cam.name}' attached to '{curve_obj.name}'.")
            except Exception as e:
                self.report({'ERROR'}, f"Camera setup failed: {e}")

        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Panel  (N-panel → Tool tab)
# ---------------------------------------------------------------------------

class CURVEGEN_PT_panel(Panel):
    bl_label = "Flight Curve Generator"
    bl_idname = "CURVEGEN_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tool"

    def draw(self, context):
        layout = self.layout
        props = context.scene.curve_gen_props

        layout.prop(props, "curve_name")
        layout.separator()

        col = layout.column(align=True)
        col.label(text="Curve Shape:")
        col.prop(props, "num_points")
        col.prop(props, "start_height_fraction")
        layout.separator()

        col = layout.column(align=True)
        col.label(text="Collision Resolution:")
        col.prop(props, "lift_fraction")
        col.prop(props, "min_clearance_fraction")
        col.prop(props, "max_collision_iters")
        layout.separator()

        # ── Camera ────────────────────────────────────────────────────────
        box = layout.box()
        row = box.row()
        row.prop(props, "attach_camera", icon='CAMERA_DATA')

        if props.attach_camera:
            col = box.column(align=True)
            col.prop(props, "use_existing_camera")
            col.prop(props, "animate_path")
            col.separator()
            col.prop(props, "look_at_mode")

            if props.look_at_mode == 'POINT':
                col.prop(props, "look_at_point", text="")
            elif props.look_at_mode == 'OBJECT':
                col.prop(props, "look_at_object", text="")

        layout.separator()
        layout.operator("curvegen.generate", icon='CURVE_DATA')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    CurveGenProperties,
    CURVEGEN_OT_generate,
    CURVEGEN_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.curve_gen_props = bpy.props.PointerProperty(type=CurveGenProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.curve_gen_props


if __name__ == "__main__":
    register()
