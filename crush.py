import bpy
import os
from mathutils import Vector

# ------------------------------
# USER SETTINGS
# ------------------------------

OBJ_PATH = r"/home/alex/projects/trash_proc/assets_copy/can/010_potted_meat_can/google_512k/textured.obj"
IMPORT_SCALE = 1.0
CLEAR_SCENE = True

FRAME_START       = 1
FRAME_PRESS_START = 10
FRAME_PRESS_END   = 40
FRAME_END         = 80

PLATE_MARGIN_XY_FACTOR = -0.5
PLATE_THICKNESS_FACTOR = 0.25
TOP_EXTRA_GAP_FACTOR   = 0.7
CRUSH_DEPTH_FACTOR     = 0.75

CAN_DECIMATE_RATIO   = 0.1   # lower = safer / lighter
CLOTH_QUALITY        = 4     # low but stable
USE_SELF_COLLISION   = False # turn on later if stable
SHELL_THICKNESS      = 0.003 # 3 mm


# ------------------------------
# UTILS
# ------------------------------

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh, do_unlink=True)


def import_obj_multi(path, scale=1.0):
    if not os.path.exists(path):
        raise FileNotFoundError(f"OBJ file not found: {path}")

    before = set(bpy.data.objects)
    res = bpy.ops.wm.obj_import(filepath=path)
    if 'CANCELLED' in res:
        raise RuntimeError(f"OBJ import cancelled: {path}")

    after = set(bpy.data.objects)
    new_objs = [o for o in (after - before) if o.type == 'MESH']
    if not new_objs:
        raise RuntimeError("No mesh objects imported from OBJ.")

    for obj in new_objs:
        obj.scale = (scale, scale, scale)

    bpy.context.view_layer.update()
    return new_objs


def get_combined_world_bbox(objs):
    xs, ys, zs = [], [], []
    for obj in objs:
        mat = obj.matrix_world
        for corner in obj.bound_box:
            p = mat @ Vector(corner)
            xs.append(p.x); ys.append(p.y); zs.append(p.z)
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def set_collision(obj):
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')
    for m in [m for m in obj.modifiers if m.type == 'COLLISION']:
        obj.modifiers.remove(m)
    obj.modifiers.new("Collision", "COLLISION")
    obj.hide_set(False)
    obj.hide_viewport = False
    obj.hide_render = False

    # Apply scale for stable collision
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def animate_press(obj, start_frame, end_frame, z_start, z_end):
    obj.location.z = z_start
    obj.keyframe_insert(data_path="location", frame=start_frame)
    obj.location.z = z_start
    obj.keyframe_insert(data_path="location", frame=FRAME_PRESS_START)
    obj.location.z = z_end
    obj.keyframe_insert(data_path="location", frame=end_frame)
    obj.location.z = z_end
    obj.keyframe_insert(data_path="location", frame=end_frame + 10)

    if obj.animation_data and obj.animation_data.action:
        for fcurve in obj.animation_data.action.fcurves:
            for kp in fcurve.keyframe_points:
                kp.interpolation = 'LINEAR'


def join_objects(objs, name="CrushCanHigh"):
    if not objs:
        return None
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')

    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    joined.name = name

    # Apply scale for cleanliness
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return joined


def create_sim_proxy(high_obj, shell_thickness=0.003, decimate_ratio=0.1):
    if high_obj is None:
        return None

    # Duplicate object + mesh
    sim = high_obj.copy()
    sim.data = high_obj.data.copy()
    sim.name = "CrushCanSim"
    bpy.context.collection.objects.link(sim)

    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    sim.select_set(True)
    bpy.context.view_layer.objects.active = sim

    # 1) Make a clean thin shell – Solidify
    solid = sim.modifiers.new("SimSolidify", 'SOLIDIFY')
    solid.thickness = shell_thickness
    solid.offset = 0.0   # shell around existing can surface
    bpy.ops.object.modifier_apply(modifier=solid.name)

    # 2) Decimate to keep it light
    dec = sim.modifiers.new("SimDecimate", 'DECIMATE')
    dec.ratio = decimate_ratio
    dec.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=dec.name)

    # Apply transforms
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    return sim


def setup_cloth_for_sim(obj):
    if obj is None:
        return
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')

    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Remove any old cloth/softbody
    for m in list(obj.modifiers):
        if m.type in {'CLOTH', 'SOFT_BODY'}:
            obj.modifiers.remove(m)

    cloth = obj.modifiers.new("CanCloth", 'CLOTH')
    cs = cloth.settings

    # --- METAL-LIKE SETTINGS ---
    cs.quality = CLOTH_QUALITY         # keep this low-ish for speed
    cs.mass = 0.1                      # lighter = less sag under gravity

    # Stiffness: crank these up so it holds its shape
    cs.tension_stiffness = 80.0        # was 15
    cs.compression_stiffness = 80.0
    cs.shear_stiffness = 40.0
    cs.bending_stiffness = 5.0         # higher = less bending

    # No internal pressure for now
    cs.use_pressure = False
    cs.uniform_pressure_force = 0.0

    # Damping to avoid jitter/explosions
    if hasattr(cs, "air_damping"):
        cs.air_damping = 2.0           # some damping, not crazy high

    # REDUCE GRAVITY EFFECT so it doesn’t crush itself
    if hasattr(cs, "effector_weights"):
        eff = cs.effector_weights
        # 1.0 = full scene gravity, 0.0 = no gravity
        eff.gravity = 0.2              # tweak between 0.0–0.5 to taste

    # Collisions
    coll = cloth.collision_settings
    coll.use_collision = True
    coll.use_self_collision = USE_SELF_COLLISION
    coll.distance_min = 0.001
    coll.self_distance_min = 0.001
    coll.friction = 5.0

    # Apply transforms for stability
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)



def bind_surface_deform(high_obj, sim_obj):
    if high_obj is None or sim_obj is None:
        return

    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')

    surf = high_obj.modifiers.new("CanSurfaceDeform", 'SURFACE_DEFORM')
    surf.target = sim_obj

    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    high_obj.select_set(True)
    sim_obj.select_set(True)
    bpy.context.view_layer.objects.active = high_obj

    bpy.ops.object.surfacedeform_bind(modifier=surf.name)

    # Hide sim from render (you can hide in viewport too if you like)
    sim_obj.hide_render = True


# ------------------------------
# MAIN
# ------------------------------

def main():
    if CLEAR_SCENE:
        clear_scene()

    scene = bpy.context.scene
    scene.frame_start = FRAME_START
    scene.frame_end = FRAME_END

    imported_objs = import_obj_multi(OBJ_PATH, scale=IMPORT_SCALE)

    # Center in XY
    min_x, max_x, min_y, max_y, min_z, max_z = get_combined_world_bbox(imported_objs)
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5

    dx = -center_x
    dy = -center_y
    for obj in imported_objs:
        obj.location.x += dx
        obj.location.y += dy

    bpy.context.view_layer.update()

    # Join into a single high-poly can
    high_can = join_objects(imported_objs, name="CrushCanHigh")

    # Recompute bbox
    min_x, max_x, min_y, max_y, min_z, max_z = get_combined_world_bbox([high_can])
    obj_width_x  = max_x - min_x
    obj_width_y  = max_y - min_y
    obj_height_z = max_z - min_z if (max_z - min_z) != 0 else 0.1

    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5

    plate_size_x = obj_width_x * (1.0 + PLATE_MARGIN_XY_FACTOR)
    plate_size_y = obj_width_y * (1.0 + PLATE_MARGIN_XY_FACTOR)

    plate_thickness = max(obj_height_z * PLATE_THICKNESS_FACTOR, obj_height_z * 0.1)

    # IMPORTANT: give some breathing room above and below the can
    clearance = obj_height_z * 0.2

    bottom_z    = min_z - plate_thickness * 0.5 - clearance
    top_start_z = max_z + plate_thickness * 0.5 + clearance + obj_height_z * TOP_EXTRA_GAP_FACTOR

    # Don't let the press intersect the bottom; stop near the can bottom
    crush_range = (max_z - bottom_z) * CRUSH_DEPTH_FACTOR
    top_end_z   = max(
        bottom_z + plate_thickness + SHELL_THICKNESS,  # <-- FIXED
        top_start_z - crush_range
    )

    print(f"Can Z: {min_z:.3f} .. {max_z:.3f}")
    print(f"Bottom plate Z: {bottom_z:.3f}")
    print(f"Top start Z: {top_start_z:.3f}, top end Z: {top_end_z:.3f}")

    # Bottom press
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(center_x, center_y, bottom_z))
    bottom_cube = bpy.context.active_object
    bottom_cube.name = "BottomPress"
    bottom_cube.scale = (
        plate_size_x * 0.5,
        plate_size_y * 0.5,
        plate_thickness * 0.5,
    )
    set_collision(bottom_cube)

    # Top press
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(center_x, center_y, top_start_z))
    top_cube = bpy.context.active_object
    top_cube.name = "TopPress"
    top_cube.scale = (
        plate_size_x * 0.5,
        plate_size_y * 0.5,
        plate_thickness * 0.5,
    )
    set_collision(top_cube)

    animate_press(
        top_cube,
        start_frame=FRAME_START,
        end_frame=FRAME_PRESS_END,
        z_start=top_start_z,
        z_end=top_end_z,
    )

    # Create sim proxy + cloth
    sim_can = create_sim_proxy(high_can, shell_thickness=SHELL_THICKNESS, decimate_ratio=CAN_DECIMATE_RATIO)
    setup_cloth_for_sim(sim_can)

    # Move sim_can to same location/orientation as high_can (should already match)
    sim_can.matrix_world = high_can.matrix_world.copy()

    # Bind high poly to sim
    bind_surface_deform(high_can, sim_can)

    scene.frame_set(FRAME_START)
    print("Setup done. Play animation or bake cloth cache. If it still explodes, lower CAN_DECIMATE_RATIO and/or CRUSH_DEPTH_FACTOR.")


if __name__ == "__main__":
    main()
