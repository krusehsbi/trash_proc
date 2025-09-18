import blenderproc as bproc

# scale_room_ready.py
# Usage:
# blenderproc run scale_room_ready.py --input_dir /path/in --out_dir /path/out --scales /path/scales.yml --floor --auto_units --seed 42
#
# The --scales file can be YAML or JSON.
# Rule keys = subfolder prefixes (relative to input_dir), e.g. "bottle", "cup", "carton".
# Supported per-category rule fields:
#   - target_size: float              # desired size for the chosen measure
#   - factor: float                   # multiplicative override (use instead of target_size)
#   - measure: "max"|"x"|"y"|"z"      # what to measure (default: "max")
#   - longest_to_z: bool              # rotate so longest axis becomes Z (upright)
#   - trim_pct: float [0..0.5)        # ignore tiny caps top/bottom when measuring z
#   - diameter_cap: float             # clamp max(X,Y) after main scaling (only shrinks)
#   - jitter_pct: float               # ± random variation on the scale factor (0.10 = ±10%)
#   - fixed_rot_deg: [rx, ry, rz]     # apply fixed rotation (degrees) before uprighting
#
# Global CLI flags:
#   --floor                           # place min Z at 0
#   --auto_units                      # crude cm<->m sanity if wildly off
#   --seed INT                        # RNG seed for jitter

import os
import argparse
import json
import math
import numpy as np
import bpy

try:
    import yaml
except Exception:
    yaml = None

# ------------------------------
# CLI
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--scales", required=True, help="Path to YAML or JSON rules")
parser.add_argument("--floor", action="store_true")
parser.add_argument("--auto_units", action="store_true", help="Auto-fix obvious cm/m mistakes")
parser.add_argument("--seed", type=int, default=42, help="Random seed for jitter")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
SUPPORTED_EXTS = (".obj", ".blend")
rng = np.random.default_rng(args.seed)

# ------------------------------
# Load rules
# ------------------------------
def load_rules(path):
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if ext in (".yml", ".yaml"):
            if yaml is None:
                raise RuntimeError("PyYAML not installed; use JSON or install pyyaml.")
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Rules file must be a dict at top level.")
    # Normalize keys (strip trailing slashes)
    out = {}
    for k, v in data.items():
        if not isinstance(v, dict):
            raise ValueError(f"Rule for '{k}' must be an object with fields.")
        out[k.strip("/")] = v
    return out

RULES = load_rules(args.scales)

def match_rule(rel_dir):
    """Return most specific matching rule by folder prefix (or default)."""
    rel_dir = rel_dir.strip("/")
    parts = rel_dir.split(os.sep) if rel_dir else []
    for i in range(len(parts), -1, -1):
        key = "/".join(parts[:i]).strip("/")
        if key and key in RULES:
            return RULES[key]
    return RULES.get("default", {"target_size": 1.0, "measure": "max"})

# ------------------------------
# Bounds & transforms
# ------------------------------
def object_world_bounds(o):
    """(bmin, bmax) WORLD space, robust across BlenderProc versions."""
    if hasattr(o, "get_bounds"):
        try:
            return o.get_bounds()
        except Exception:
            pass
    bb = np.asarray(o.get_bound_box())  # (8,3) local
    if hasattr(o, "get_local2world_mat"):
        M = np.asarray(o.get_local2world_mat())  # 4x4
        bb_h = np.concatenate([bb, np.ones((bb.shape[0], 1))], axis=1)
        bb_w = (M @ bb_h.T).T[:, :3]
    else:
        bb_w = bb
    return bb_w.min(axis=0), bb_w.max(axis=0)

def combined_bounds(objs):
    mins = np.array([np.inf, np.inf, np.inf], dtype=float)
    maxs = -mins
    for o in objs:
        bmin, bmax = object_world_bounds(o)
        mins = np.minimum(mins, bmin)
        maxs = np.maximum(maxs, bmax)
    return mins, maxs

def extent_xyz(objs):
    bmin, bmax = combined_bounds(objs)
    ext = bmax - bmin
    return float(ext[0]), float(ext[1]), float(ext[2])

def recenter(objs):
    bmin, bmax = combined_bounds(objs)
    center = 0.5 * (bmin + bmax)
    for o in objs:
        o.set_location(o.get_location() - center)

def floor_place(objs):
    bmin, bmax = combined_bounds(objs)
    dz = -float(bmin[2])
    if dz != 0.0:
        for o in objs:
            loc = o.get_location()
            o.set_location([loc[0], loc[1], loc[2] + dz])

def rotate_euler_deg(objs, rx, ry, rz):
    rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
    for o in objs:
        r = o.get_rotation_euler()
        o.set_rotation_euler([r[0] + rx, r[1] + ry, r[2] + rz])

def rotate_longest_axis_to_z(objs):
    """Try discrete 90° rotations to make Z the longest extent."""
    # Snapshot original rotations
    orig_rots = [o.get_rotation_euler().copy() for o in objs]

    def measure_after(op=None):
        for o, r in zip(objs, orig_rots):
            o.set_rotation_euler(r)
        if op:
            op()
        # recentre before measuring (stable bounds)
        recenter(objs)
        _, _, z = extent_xyz(objs)
        return z

    candidates = [
        (None, lambda: None),
        ("x+90", lambda: rotate_euler_deg(objs, +90, 0, 0)),
        ("x-90", lambda: rotate_euler_deg(objs, -90, 0, 0)),
        ("y+90", lambda: rotate_euler_deg(objs, 0, +90, 0)),
        ("y-90", lambda: rotate_euler_deg(objs, 0, -90, 0)),
    ]
    best = max(candidates, key=lambda c: measure_after(c[1]))
    # Apply best for real
    for o, r in zip(objs, orig_rots):
        o.set_rotation_euler(r)
    if best[1] is not None:
        best[1]()  # apply

def measure_size(objs, measure="max", trim_pct=0.0):
    bmin, bmax = combined_bounds(objs)
    ext = bmax - bmin
    if measure == "z":
        z = float(ext[2])
        if trim_pct and 0.0 < float(trim_pct) < 0.5:
            return max(0.0, z * (1.0 - 2.0 * float(trim_pct)))
        return z
    if measure == "x":
        return float(ext[0])
    if measure == "y":
        return float(ext[1])
    return float(np.max(ext))

def xy_diameter(objs):
    bmin, bmax = combined_bounds(objs)
    ext = bmax - bmin
    return float(max(ext[0], ext[1]))

# ------------------------------
# Export helpers (bpy)
# ------------------------------
def _bpy_objects_from_meshobjects(mesh_objs):
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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path, copy=True)

def export_selected_as_obj(out_path, selected_bpy_objects):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for ob in bpy.data.objects:
        ob.select_set(False)
    for ob in selected_bpy_objects:
        ob.select_set(True)
    if selected_bpy_objects:
        bpy.context.view_layer.objects.active = selected_bpy_objects[0]
    # Blender 4.x operator
    try:
        bpy.ops.wm.obj_export(
            filepath=out_path,
            export_selected_objects=True,
            apply_modifiers=False,
            export_materials=True,
            path_mode='AUTO',
        )
        return True
    except Exception:
        pass
    # Legacy operator
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
# Core processing
# ------------------------------
def normalize_and_export(in_path, out_path, rule):
    # Clean scene before each import
    bproc.clean_up()

    ext = os.path.splitext(in_path)[1].lower()
    print(f"--> Processing: {in_path}")
    print(f"    Rule: {rule}")

    # Load
    if ext == ".obj":
        objs = bproc.loader.load_obj(in_path)
    elif ext == ".blend":
        objs = bproc.loader.load_blend(in_path)
        objs = [o for o in objs if isinstance(o, bproc.types.MeshObject)]
        if not objs:
            objs = [o for o in bproc.scene.get_objects() if isinstance(o, bproc.types.MeshObject)]
    else:
        print(f"    Skipping unsupported: {in_path}")
        return False

    if not objs:
        print("    No mesh objects found, skipping.")
        return False

    # Optional fixed rotation BEFORE uprighting (degrees)
    fix_rot = rule.get("fixed_rot_deg", None)
    if isinstance(fix_rot, (list, tuple)) and len(fix_rot) == 3:
        rotate_euler_deg(objs, float(fix_rot[0]), float(fix_rot[1]), float(fix_rot[2]))

    # Upright: rotate to make the longest axis vertical if requested
    if bool(rule.get("longest_to_z", False)):
        rotate_longest_axis_to_z(objs)

    # --- Auto units sanity (crude cm<->m fix) ---
    if args.auto_units:
        x, y, z = extent_xyz(objs)
        max_dim = max(x, y, z)
        if max_dim > 50.0:           # probably centimeters read as meters
            unit_fix = 0.01
        elif 0 < max_dim < 0.01:     # probably meters read as centimeters
            unit_fix = 100.0
        else:
            unit_fix = 1.0
        if unit_fix != 1.0:
            for o in objs:
                o.set_scale(o.get_scale() * unit_fix)

    # Measure current size with chosen metric
    measure_key = str(rule.get("measure", "max")).lower()
    trim_pct = float(rule.get("trim_pct", 0.0))
    cur_size = measure_size(objs, measure=measure_key, trim_pct=trim_pct)
    if not np.isfinite(cur_size) or cur_size <= 0:
        print(f"    Non-positive measured size (measure={measure_key}, trim={trim_pct}): {cur_size}")
        return False

    # Compute base factor
    if "target_size" in rule:
        factor = float(rule["target_size"]) / cur_size
    elif "factor" in rule:
        factor = float(rule["factor"])
    else:
        factor = 1.0

    # Jitter (± jitter_pct)
    j = float(rule.get("jitter_pct", 0.0))
    if j > 0:
        factor *= float(rng.uniform(1.0 - j, 1.0 + j))

    # Apply main scaling
    if factor != 1.0:
        for o in objs:
            o.set_scale(o.get_scale() * factor)

    # Optional diameter cap after main scaling
    if "diameter_cap" in rule:
        cap = float(rule["diameter_cap"])
        cur_diam = xy_diameter(objs)
        if np.isfinite(cur_diam) and cur_diam > cap > 0:
            shrink = cap / cur_diam
            for o in objs:
                o.set_scale(o.get_scale() * shrink)

    # Recenter to origin
    recenter(objs)

    # Optional: put on floor (min Z = 0)
    if args.floor:
        floor_place(objs)

    # Export
    try:
        if ext == ".obj":
            bpy_objs = _bpy_objects_from_meshobjects(objs)
            if not bpy_objs:
                print("    Could not resolve bpy objects by name, skipping export.")
                return False
            ok = export_selected_as_obj(out_path, bpy_objs)
            if not ok:
                return False
        else:
            save_as_blend(out_path)
    except Exception as e:
        print(f"    Error saving {out_path}: {e}")
        return False

    print(f"    Saved: {out_path}")
    return True

# ------------------------------
# Walk & apply to all files
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

        rule = match_rule(rel_dir)
        out_dir_full = os.path.join(args.out_dir, rel_dir)
        out_path = os.path.join(out_dir_full, f"{name}_normalized{ext}")

        processed += 1
        if normalize_and_export(in_path, out_path, rule):
            saved += 1

print(f"Done. Processed {processed} file(s), saved {saved}.")
import blenderproc as bproc

# scale_room_ready.py
# Usage:
# blenderproc run scale_room_ready.py --input_dir /path/in --out_dir /path/out --scales /path/scales.yml --floor --auto_units --seed 42
#
# The --scales file can be YAML or JSON.
# Rule keys = subfolder prefixes (relative to input_dir), e.g. "bottle", "cup", "carton".
# Supported per-category rule fields:
#   - target_size: float              # desired size for the chosen measure
#   - factor: float                   # multiplicative override (use instead of target_size)
#   - measure: "max"|"x"|"y"|"z"      # what to measure (default: "max")
#   - longest_to_z: bool              # rotate so longest axis becomes Z (upright)
#   - trim_pct: float [0..0.5)        # ignore tiny caps top/bottom when measuring z
#   - diameter_cap: float             # clamp max(X,Y) after main scaling (only shrinks)
#   - jitter_pct: float               # ± random variation on the scale factor (0.10 = ±10%)
#   - fixed_rot_deg: [rx, ry, rz]     # apply fixed rotation (degrees) before uprighting
#
# Global CLI flags:
#   --floor                           # place min Z at 0
#   --auto_units                      # crude cm<->m sanity if wildly off
#   --seed INT                        # RNG seed for jitter

import os
import argparse
import json
import math
import numpy as np
import bpy

try:
    import yaml
except Exception:
    yaml = None

# ------------------------------
# CLI
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--scales", required=True, help="Path to YAML or JSON rules")
parser.add_argument("--floor", action="store_true")
parser.add_argument("--auto_units", action="store_true", help="Auto-fix obvious cm/m mistakes")
parser.add_argument("--seed", type=int, default=42, help="Random seed for jitter")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
SUPPORTED_EXTS = (".obj", ".blend")
rng = np.random.default_rng(args.seed)

# ------------------------------
# Load rules
# ------------------------------
def load_rules(path):
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if ext in (".yml", ".yaml"):
            if yaml is None:
                raise RuntimeError("PyYAML not installed; use JSON or install pyyaml.")
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Rules file must be a dict at top level.")
    # Normalize keys (strip trailing slashes)
    out = {}
    for k, v in data.items():
        if not isinstance(v, dict):
            raise ValueError(f"Rule for '{k}' must be an object with fields.")
        out[k.strip("/")] = v
    return out

RULES = load_rules(args.scales)

def match_rule(rel_dir):
    """Return most specific matching rule by folder prefix (or default)."""
    rel_dir = rel_dir.strip("/")
    parts = rel_dir.split(os.sep) if rel_dir else []
    for i in range(len(parts), -1, -1):
        key = "/".join(parts[:i]).strip("/")
        if key and key in RULES:
            return RULES[key]
    return RULES.get("default", {"target_size": 1.0, "measure": "max"})

# ------------------------------
# Bounds & transforms
# ------------------------------
def object_world_bounds(o):
    """(bmin, bmax) WORLD space, robust across BlenderProc versions."""
    if hasattr(o, "get_bounds"):
        try:
            return o.get_bounds()
        except Exception:
            pass
    bb = np.asarray(o.get_bound_box())  # (8,3) local
    if hasattr(o, "get_local2world_mat"):
        M = np.asarray(o.get_local2world_mat())  # 4x4
        bb_h = np.concatenate([bb, np.ones((bb.shape[0], 1))], axis=1)
        bb_w = (M @ bb_h.T).T[:, :3]
    else:
        bb_w = bb
    return bb_w.min(axis=0), bb_w.max(axis=0)

def combined_bounds(objs):
    mins = np.array([np.inf, np.inf, np.inf], dtype=float)
    maxs = -mins
    for o in objs:
        bmin, bmax = object_world_bounds(o)
        mins = np.minimum(mins, bmin)
        maxs = np.maximum(maxs, bmax)
    return mins, maxs

def extent_xyz(objs):
    bmin, bmax = combined_bounds(objs)
    ext = bmax - bmin
    return float(ext[0]), float(ext[1]), float(ext[2])

def recenter(objs):
    bmin, bmax = combined_bounds(objs)
    center = 0.5 * (bmin + bmax)
    for o in objs:
        o.set_location(o.get_location() - center)

def floor_place(objs):
    bmin, bmax = combined_bounds(objs)
    dz = -float(bmin[2])
    if dz != 0.0:
        for o in objs:
            loc = o.get_location()
            o.set_location([loc[0], loc[1], loc[2] + dz])

def rotate_euler_deg(objs, rx, ry, rz):
    rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
    for o in objs:
        r = o.get_rotation_euler()
        o.set_rotation_euler([r[0] + rx, r[1] + ry, r[2] + rz])

def rotate_longest_axis_to_z(objs):
    """Try discrete 90° rotations to make Z the longest extent."""
    # Snapshot original rotations
    orig_rots = [o.get_rotation_euler().copy() for o in objs]

    def measure_after(op=None):
        for o, r in zip(objs, orig_rots):
            o.set_rotation_euler(r)
        if op:
            op()
        # recentre before measuring (stable bounds)
        recenter(objs)
        _, _, z = extent_xyz(objs)
        return z

    candidates = [
        (None, lambda: None),
        ("x+90", lambda: rotate_euler_deg(objs, +90, 0, 0)),
        ("x-90", lambda: rotate_euler_deg(objs, -90, 0, 0)),
        ("y+90", lambda: rotate_euler_deg(objs, 0, +90, 0)),
        ("y-90", lambda: rotate_euler_deg(objs, 0, -90, 0)),
    ]
    best = max(candidates, key=lambda c: measure_after(c[1]))
    # Apply best for real
    for o, r in zip(objs, orig_rots):
        o.set_rotation_euler(r)
    if best[1] is not None:
        best[1]()  # apply

def measure_size(objs, measure="max", trim_pct=0.0):
    bmin, bmax = combined_bounds(objs)
    ext = bmax - bmin
    if measure == "z":
        z = float(ext[2])
        if trim_pct and 0.0 < float(trim_pct) < 0.5:
            return max(0.0, z * (1.0 - 2.0 * float(trim_pct)))
        return z
    if measure == "x":
        return float(ext[0])
    if measure == "y":
        return float(ext[1])
    return float(np.max(ext))

def xy_diameter(objs):
    bmin, bmax = combined_bounds(objs)
    ext = bmax - bmin
    return float(max(ext[0], ext[1]))

# ------------------------------
# Export helpers (bpy)
# ------------------------------
def _bpy_objects_from_meshobjects(mesh_objs):
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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path, copy=True)

def export_selected_as_obj(out_path, selected_bpy_objects):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for ob in bpy.data.objects:
        ob.select_set(False)
    for ob in selected_bpy_objects:
        ob.select_set(True)
    if selected_bpy_objects:
        bpy.context.view_layer.objects.active = selected_bpy_objects[0]
    # Blender 4.x operator
    try:
        bpy.ops.wm.obj_export(
            filepath=out_path,
            export_selected_objects=True,
            apply_modifiers=False,
            export_materials=True,
            path_mode='AUTO',
        )
        return True
    except Exception:
        pass
    # Legacy operator
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
# Core processing
# ------------------------------
def normalize_and_export(in_path, out_path, rule):
    # Clean scene before each import
    bproc.clean_up()

    ext = os.path.splitext(in_path)[1].lower()
    print(f"--> Processing: {in_path}")
    print(f"    Rule: {rule}")

    # Load
    if ext == ".obj":
        objs = bproc.loader.load_obj(in_path)
    elif ext == ".blend":
        objs = bproc.loader.load_blend(in_path)
        objs = [o for o in objs if isinstance(o, bproc.types.MeshObject)]
        if not objs:
            objs = [o for o in bproc.scene.get_objects() if isinstance(o, bproc.types.MeshObject)]
    else:
        print(f"    Skipping unsupported: {in_path}")
        return False

    if not objs:
        print("    No mesh objects found, skipping.")
        return False

    # Optional fixed rotation BEFORE uprighting (degrees)
    fix_rot = rule.get("fixed_rot_deg", None)
    if isinstance(fix_rot, (list, tuple)) and len(fix_rot) == 3:
        rotate_euler_deg(objs, float(fix_rot[0]), float(fix_rot[1]), float(fix_rot[2]))

    # Upright: rotate to make the longest axis vertical if requested
    if bool(rule.get("longest_to_z", False)):
        rotate_longest_axis_to_z(objs)

    # --- Auto units sanity (crude cm<->m fix) ---
    if args.auto_units:
        x, y, z = extent_xyz(objs)
        max_dim = max(x, y, z)
        if max_dim > 50.0:           # probably centimeters read as meters
            unit_fix = 0.01
        elif 0 < max_dim < 0.01:     # probably meters read as centimeters
            unit_fix = 100.0
        else:
            unit_fix = 1.0
        if unit_fix != 1.0:
            for o in objs:
                o.set_scale(o.get_scale() * unit_fix)

    # Measure current size with chosen metric
    measure_key = str(rule.get("measure", "max")).lower()
    trim_pct = float(rule.get("trim_pct", 0.0))
    cur_size = measure_size(objs, measure=measure_key, trim_pct=trim_pct)
    if not np.isfinite(cur_size) or cur_size <= 0:
        print(f"    Non-positive measured size (measure={measure_key}, trim={trim_pct}): {cur_size}")
        return False

    # Compute base factor
    if "target_size" in rule:
        factor = float(rule["target_size"]) / cur_size
    elif "factor" in rule:
        factor = float(rule["factor"])
    else:
        factor = 1.0

    # Jitter (± jitter_pct)
    j = float(rule.get("jitter_pct", 0.0))
    if j > 0:
        factor *= float(rng.uniform(1.0 - j, 1.0 + j))

    # Apply main scaling
    if factor != 1.0:
        for o in objs:
            o.set_scale(o.get_scale() * factor)

    # Optional diameter cap after main scaling
    if "diameter_cap" in rule:
        cap = float(rule["diameter_cap"])
        cur_diam = xy_diameter(objs)
        if np.isfinite(cur_diam) and cur_diam > cap > 0:
            shrink = cap / cur_diam
            for o in objs:
                o.set_scale(o.get_scale() * shrink)

    # Recenter to origin
    recenter(objs)

    # Optional: put on floor (min Z = 0)
    if args.floor:
        floor_place(objs)

    # Export
    try:
        if ext == ".obj":
            bpy_objs = _bpy_objects_from_meshobjects(objs)
            if not bpy_objs:
                print("    Could not resolve bpy objects by name, skipping export.")
                return False
            ok = export_selected_as_obj(out_path, bpy_objs)
            if not ok:
                return False
        else:
            save_as_blend(out_path)
    except Exception as e:
        print(f"    Error saving {out_path}: {e}")
        return False

    print(f"    Saved: {out_path}")
    return True

# ------------------------------
# Walk & apply to all files
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

        rule = match_rule(rel_dir)
        out_dir_full = os.path.join(args.out_dir, rel_dir)
        out_path = os.path.join(out_dir_full, f"{name}_normalized{ext}")

        processed += 1
        if normalize_and_export(in_path, out_path, rule):
            saved += 1

print(f"Done. Processed {processed} file(s), saved {saved}.")
