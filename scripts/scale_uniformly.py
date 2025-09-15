import blenderproc as bproc

# normalize_with_blenderproc.py
# Usage:
# blenderproc run normalize_with_blenderproc.py --input_dir /path/in --out_dir /path/out --target_size 1.0 --floor

import os
import argparse
import numpy as np
import bpy  # Use Blender's native exporters

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--target_size", type=float, default=1.0)
parser.add_argument("--floor", action="store_true")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

SUPPORTED_EXTS = (".obj", ".blend")

# ------------------------------
# Bounds helpers (version-proof)
# ------------------------------
def object_world_bounds(o):
    """Return (bmin, bmax) in WORLD space, across BlenderProc versions."""
    if hasattr(o, "get_bounds"):
        try:
            return o.get_bounds()  # -> (3,), (3,)
        except Exception:
            pass

    # Fallback: transform 8 local bound-box corners by local2world
    bb = np.asarray(o.get_bound_box())  # (8, 3) local space
    if hasattr(o, "get_local2world_mat"):
        M = np.asarray(o.get_local2world_mat())  # 4x4
        bb_h = np.concatenate([bb, np.ones((bb.shape[0], 1))], axis=1)  # (8,4)
        bb_w = (M @ bb_h.T).T[:, :3]
    else:
        bb_w = bb  # best effort
    return bb_w.min(axis=0), bb_w.max(axis=0)

def combined_bounds(objs):
    mins = np.array([np.inf, np.inf, np.inf], dtype=float)
    maxs = -mins
    for o in objs:
        bmin, bmax = object_world_bounds(o)
        mins = np.minimum(mins, bmin)
        maxs = np.maximum(maxs, bmax)
    return mins, maxs

# ------------------------------
# Export helpers (bpy-based)
# ------------------------------
def _bpy_objects_from_meshobjects(mesh_objs):
    """Resolve Blender (bpy) objects by name."""
    out = []
    for mo in mesh_objs:
        try:
            name = mo.get_name()
        except Exception:
            name = None
        if not name:
            continue
        bo = bpy.data.objects.get(name)
        if bo is not None:
            out.append(bo)
    return out

def save_as_blend(out_path):
    """Save current scene to a .blend file (copy, do not replace current)."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Using copy=True avoids changing the 'current' blend file
    bpy.ops.wm.save_as_mainfile(filepath=out_path, copy=True)

def export_selected_as_obj(out_path, selected_bpy_objects):
    """Export given bpy objects to OBJ at out_path."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Deselect everything
    for ob in bpy.data.objects:
        ob.select_set(False)

    # Select only the ones we want and set an active object
    for ob in selected_bpy_objects:
        ob.select_set(True)
    if selected_bpy_objects:
        bpy.context.view_layer.objects.active = selected_bpy_objects[0]

    # Try Blender 4.x operator first
    try:
        bpy.ops.wm.obj_export(
            filepath=out_path,
            export_selected_objects=True,
            apply_modifiers=False,
            export_materials=True,
            path_mode='AUTO',  # don't rewrite paths
        )
        return True
    except Exception:
        pass

    # Fallback for Blender 2.8x/3.x legacy operator
    try:
        bpy.ops.export_scene.obj(
            filepath=out_path,
            use_selection=True,
            use_mesh_modifiers=False,
            use_materials=True,
            path_mode='AUTO',
        )
        return True
    except Exception as e:
        print(f"    OBJ export failed: {e}")
        return False

# ------------------------------
# Core normalization
# ------------------------------
def normalize_file(in_path, out_path):
    # Reset scene each file for a clean import
    bproc.clean_up()

    ext = os.path.splitext(in_path)[1].lower()
    print(f"--> Processing: {in_path}")

    # Load
    if ext == ".obj":
        objs = bproc.loader.load_obj(in_path)
    elif ext == ".blend":
        objs = bproc.loader.load_blend(in_path)
        # Keep only meshes (robust)
        objs = [o for o in objs if isinstance(o, bproc.types.MeshObject)]
        if not objs:
            objs = [o for o in bproc.scene.get_objects() if isinstance(o, bproc.types.MeshObject)]
    else:
        print(f"    Skipping unsupported: {in_path}")
        return False

    if not objs:
        print("    No mesh objects found, skipping.")
        return False

    # Compute current size
    bmin, bmax = combined_bounds(objs)
    size = bmax - bmin
    max_dim = float(np.max(size))
    if not np.isfinite(max_dim) or max_dim <= 0:
        print(f"    Non-positive size (max_dim={max_dim}), skipping.")
        return False

    # Uniform scale to target_size on the largest dimension
    factor = args.target_size / max_dim
    for o in objs:
        o.set_scale(o.get_scale() * factor)

    # Recenter to origin
    bmin, bmax = combined_bounds(objs)
    center = 0.5 * (bmin + bmax)
    for o in objs:
        o.set_location(o.get_location() - center)

    # Optional: place on floor (min Z = 0)
    if args.floor:
        bmin, bmax = combined_bounds(objs)
        dz = -float(bmin[2])
        if dz != 0.0:
            for o in objs:
                loc = o.get_location()
                o.set_location([loc[0], loc[1], loc[2] + dz])

    # Save (mirror structure already handled by caller)
    try:
        if ext == ".obj":
            # Export only these objects to OBJ
            bpy_objs = _bpy_objects_from_meshobjects(objs)
            if not bpy_objs:
                print("    Could not resolve bpy objects by name, skipping export.")
                return False
            ok = export_selected_as_obj(out_path, bpy_objs)
            if not ok:
                return False
        else:
            # Save entire scene to a .blend copy
            save_as_blend(out_path)
    except Exception as e:
        print(f"    Error saving {out_path}: {e}")
        return False

    print(f"    Saved: {out_path}")
    return True

# ------------------------------
# Walk recursively and mirror dirs
# ------------------------------
processed = 0
saved = 0

for root, _, files in os.walk(args.input_dir):
    for fname in files:
        if not fname.lower().endswith(SUPPORTED_EXTS):
            continue
        in_path = os.path.join(root, fname)
        rel_path = os.path.relpath(in_path, args.input_dir)
        rel_dir, base = os.path.split(rel_path)
        name, ext = os.path.splitext(base)

        out_dir_full = os.path.join(args.out_dir, rel_dir)
        out_path = os.path.join(out_dir_full, f"{name}_normalized{ext}")

        processed += 1
        if normalize_file(in_path, out_path):
            saved += 1

print(f"Done. Processed {processed} file(s), saved {saved}.")