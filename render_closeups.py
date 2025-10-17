import blenderproc as bproc

# render_closeups.py
# Recursively render close-up PNGs of assets with solid framing.
# blenderproc run render_closeups.py --assets_dir ... --output_dir ...

import os, sys, math, argparse
import bpy
import bmesh
from mathutils import Vector, Matrix

# ------------------------------------------------------------------------------------
# Args
# ------------------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--assets_dir", required=True, type=str, help="Root folder (searched recursively).")
p.add_argument("--output_dir", required=True, type=str, help="Output root for PNGs.")
p.add_argument("--res", type=int, default=1024)
p.add_argument("--samples", type=int, default=256)
p.add_argument("--transparent", type=str, default="True")
p.add_argument("--margin", type=float, default=1.12, help=">1.0 adds padding around the object.")
p.add_argument("--elevation_deg", type=float, default=18.0)
p.add_argument("--azimuth_deg", type=float, default=35.0)
p.add_argument("--white_bg", action="store_true", help="Force white background (disables transparency).")
args, _ = p.parse_known_args()

ASSETS_DIR = os.path.abspath(args.assets_dir)
OUT_DIR    = os.path.abspath(args.output_dir)
RES        = int(args.res)
SAMPLES    = int(args.samples)
TRANSPARENT = (str(args.transparent).lower() in ["1","true","yes","y"]) and not args.white_bg
MARGIN     = max(1.01, float(args.margin))
ELEV_DEG   = float(args.elevation_deg)
AZIM_DEG   = float(args.azimuth_deg)
os.makedirs(OUT_DIR, exist_ok=True)

# Supported formats
EXTS = {".obj", ".fbx", ".gltf", ".glb", ".stl", ".ply", ".dae", ".blend"}

# ------------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------------
def purge_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for _ in range(2):
        try:
            bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        except Exception:
            pass

def gathered_new_objects(before_names):
    return [bpy.data.objects[n] for n in (set(bpy.data.objects.keys()) - set(before_names))]

def import_asset(path):
    """Import a single asset file. Returns list of newly created objects."""
    ext = os.path.splitext(path)[1].lower()
    before = list(bpy.data.objects.keys())
    try:
        if ext == ".obj":
            bpy.ops.wm.obj_import(filepath=path)
        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=path, automatic_bone_orientation=True,
                                     use_image_search=True, use_custom_normals=True, axis_forward='-Z', axis_up='Y')
        elif ext in [".gltf", ".glb"]:
            bpy.ops.import_scene.gltf(filepath=path, merge_vertices=True)
        elif ext == ".stl":
            bpy.ops.import_mesh.stl(filepath=path, global_scale=1.0, use_scene_unit=True)
        elif ext == ".ply":
            bpy.ops.import_mesh.ply(filepath=path)
        elif ext == ".dae":
            bpy.ops.wm.collada_import(filepath=path, auto_connect=True, fix_orientation=True)
        elif ext == ".blend":
            with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
                data_to.objects = data_from.objects
            for o in data_to.objects:
                if o is not None:
                    bpy.context.scene.collection.objects.link(o)
        else:
            return []
    except Exception as e:
        
        print(f"⚠️ Import failed: {path} -> {e}")
        return []
    imported = gathered_new_objects(before)
    # Drop imported lights/cameras/armatures/empties
    for o in list(imported):
        if o.type in {"LIGHT", "CAMERA"}:
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except Exception:
                pass
    return [o for o in imported if o.users_scene]

def make_instances_real_and_apply(objs):
    """Make instances real, apply transforms on meshes."""
    # Make instances real
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        if o.instance_type != 'NONE' or o.is_instancer:
            o.select_set(True)
    if any(o.select_get() for o in objs):
        try:
            bpy.ops.object.duplicates_make_real()
        except Exception:
            pass
    bpy.ops.object.select_all(action='DESELECT')
    # Apply transforms
    for o in list(bpy.context.scene.objects):
        if o.type in {"MESH", "CURVE", "SURFACE", "META", "FONT"}:
            try:
                bpy.context.view_layer.objects.active = o
                o.select_set(True)
                bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
                o.select_set(False)
            except Exception:
                pass

def convert_non_mesh_to_mesh(objs):
    out = []
    for o in objs:
        if o.type == "MESH":
            out.append(o)
        elif o.type in {"CURVE", "SURFACE", "META", "FONT"}:
            try:
                bpy.context.view_layer.objects.active = o
                o.select_set(True)
                bpy.ops.object.convert(target='MESH', keep_original=False)
                o.select_set(False)
            except Exception:
                pass
    # Collect again
    return [o for o in bpy.context.scene.objects if o.type == "MESH" and not o.hide_get()]

def join_meshes(meshes):
    if not meshes:
        return None
    bpy.ops.object.select_all(action='DESELECT')
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    try:
        bpy.ops.object.join()
    except Exception:
        pass
    joined = bpy.context.active_object
    return joined if joined and joined.type == "MESH" else None

def fix_geometry(obj):
    """Recalc normals outside, remove doubles, shade smooth + auto smooth."""
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    try:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    except Exception:
        pass
    bm.to_mesh(me)
    bm.free()
    obj.select_set(True)
    try:
        bpy.ops.object.shade_smooth()
    except Exception:
        pass
    obj.select_set(False)
    bpy.context.view_layer.update()

def bbox_world(obj):
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    vmin = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    vmax = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return vmin, vmax

def center_ground_normalize(obj, target_max=1.0):
    """Center at origin, put on ground (z=0), scale to target_max largest dimension."""
    vmin, vmax = bbox_world(obj)
    center = (vmin + vmax) * 0.5
    dims = vmax - vmin
    max_dim = max(dims.x, dims.y, dims.z, 1e-9)
    # Move so geometric center is at XY origin
    obj.location -= Vector((center.x, center.y, 0.0))
    bpy.context.view_layer.update()
    # Ground it: move up so bottom touches z=0
    vmin, vmax = bbox_world(obj)
    obj.location.z -= vmin.z
    bpy.context.view_layer.update()
    # Normalize scale
    scale = target_max / max_dim
    obj.scale *= scale
    bpy.context.view_layer.update()

def setup_world():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = SAMPLES
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode  = 'RGBA'
    if TRANSPARENT:
        scene.render.film_transparent = True
    else:
        world = bpy.data.worlds.get("World")
        if world is None:
            world = bpy.data.worlds.new("World")
            scene.world = world
        world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs[0].default_value = (1, 1, 1, 1)
            bg.inputs[1].default_value = 1.0

def make_three_point_lighting():
    def area(name, loc, rot_deg, power=1200, size=1.8):
        d = bpy.data.lights.new(name, 'AREA')
        d.energy = power
        d.shape = 'SQUARE'
        d.size  = size
        o = bpy.data.objects.new(name, d)
        bpy.context.scene.collection.objects.link(o)
        o.location = Vector(loc)
        o.rotation_euler = Vector([math.radians(a) for a in rot_deg])
        return o
    area("KeyLight",  (2.2, -2.2, 2.6), (65, 0,  35), power=1800, size=2.0)
    area("FillLight", (-2.0,  2.0, 1.8), (70, 0, -30), power=700,  size=1.6)
    area("RimLight",  (-1.6, -2.0, 3.2), (110,0, 150), power=900,  size=1.2)

def new_camera():
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 55.0
    cam_data.sensor_width = 36.0
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam

def look_at(eye, target, up=Vector((0,0,1))):
    f = (target - eye).normalized()
    r = f.cross(up).normalized()
    u = r.cross(f).normalized()
    rot = Matrix(((r.x, u.x, -f.x, 0),
                  (r.y, u.y, -f.y, 0),
                  (r.z, u.z, -f.z, 0),
                  (0,   0,    0,   1)))
    return Matrix.Translation(eye) @ rot

def frame_tight(cam, obj_center, half_extents, margin, aspect=1.0):
    """
    Place camera so the object fits in BOTH axes given aspect & margin.
    half_extents: Vector of half-dims (x,y,z).
    """
    # Camera direction from spherical angles
    elev = math.radians(ELEV_DEG)
    azim = math.radians(AZIM_DEG)
    # Camera FOVs
    camd = cam.data
    hfov = 2.0 * math.atan((camd.sensor_width * 0.5) / camd.lens)
    vfov = 2.0 * math.atan(math.tan(hfov * 0.5) / aspect)

    # Effective projected half-size along camera-local X (horizontal) and Y (vertical).
    # Approximate using the largest of XY half-extents for horizontal and vertical.
    hx = max(half_extents.x, half_extents.y)  # conservative
    hy = max(half_extents.z, max(half_extents.x, half_extents.y)*0.2)  # give some headroom vertically

    dist_x = (hx * margin) / math.tan(hfov * 0.5)
    dist_y = (hy * margin) / math.tan(vfov * 0.5)
    dist = max(dist_x, dist_y, 0.6)

    eye = Vector((
        obj_center.x + dist * math.cos(elev) * math.cos(azim),
        obj_center.y + dist * math.cos(elev) * math.sin(azim),
        obj_center.z + dist * math.sin(elev)
    ))
    cam.matrix_world = look_at(eye, obj_center)

def collect_files(root):
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTS:
                out.append(os.path.join(dirpath, f))
    out.sort()
    return out

# ------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------
def main():
    bproc.init()
    setup_world()

    files = collect_files(ASSETS_DIR)
    if not files:
        print(f"No supported files found under: {ASSETS_DIR}")
        sys.exit(0)

    for path in files:
        rel_dir = os.path.relpath(os.path.dirname(path), ASSETS_DIR)
        save_dir = os.path.join(OUT_DIR, rel_dir)
        os.makedirs(save_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(path))[0]
        out_png = os.path.join(save_dir, f"{name}.png")

        print(f"▶ {path}")

        # Clean scene and build neutral studio
        purge_scene()
        setup_world()
        make_three_point_lighting()
        cam = new_camera()

        # Import
        imported = import_asset(path)
        if not imported:
            print(f"   Skipping (nothing imported).")
            continue

        make_instances_real_and_apply(imported)
        meshes = convert_non_mesh_to_mesh(imported)
        meshes = [m for m in meshes if m.type == "MESH" and m.data is not None and m.visible_get()]
        if not meshes:
            print("   Skipping (no meshes).")
            continue

        joined = join_meshes(meshes)
        if not joined:
            print("   Skipping (join failed).")
            continue

        # Clean & normalize geometry
        fix_geometry(joined)
        center_ground_normalize(joined, target_max=1.0)

        # Compute framing parameters
        vmin, vmax = bbox_world(joined)
        center = (vmin + vmax) * 0.5
        dims = (vmax - vmin)
        half = dims * 0.5

        # Frame camera tightly (square aspect)
        frame_tight(cam, center, half, margin=MARGIN, aspect=1.0)

        # Render
        scn = bpy.context.scene
        scn.render.filepath = out_png
        try:
            bpy.ops.render.render(write_still=True)
            print(f"✅ Saved: {out_png}")
        except Exception as e:
            print(f"   ⚠️ Render failed: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
