import blenderproc as bproc

# scale_by_category.py
# Usage:
# blenderproc run scale_by_category.py --input_dir /path/in --out_dir /path/out --scales /path/scales.json --floor
# The --scales file can be JSON or YAML. Each key is a subfolder prefix relative to input_dir.
# Rule schema per key: { "factor": <float> } or { "target_size": <float> }. "default" is optional.

import os
import argparse
import json
import numpy as np
import bpy

try:
    import yaml  # optional, only if you pass a .yml/.yaml
except Exception:
    yaml = None

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--scales", required=True, help="Path to JSON or YAML mapping file")
parser.add_argument("--floor", action="store_true")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
SUPPORTED_EXTS = (".obj", ".blend")

# ------------------------------
# Load rules
# ------------------------------
def load_rules(path):
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if ext in [".yml", ".yaml"]:
            if yaml is None:
                raise RuntimeError("PyYAML not available; install it or use JSON for --scales")
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Scales file must be a dict at the top level")
    # Normalize rules: ensure only allowed keys
    normalized = {}
    for k, v in data.items():
        if not isinstance(v, dict):
            raise ValueError(f"Rule for '{k}' must be an object with 'factor' or 'target_size'")
        rule = {}
        if "factor" in v:
            rule["factor"] = float(v["factor"])
        if "target_size" in v:
            rule["target_size"] = float(v["target_size"])
        if not rule:
            raise ValueError(f"Rule for '{k}' must contain 'factor' or 'target_size'")
        normalized[k.strip("/")] = rule
    return normalized

RULES = load_rules(args.scales)

def match_rule(rel_dir):
    """Return the most specific rule matching rel_dir (subfolder path)."""
    rel_dir = rel_dir.strip("/")

    # Candidates: exact rel_dir and all its prefixes, plus 'default'
    parts = rel_dir.split(os.sep) if rel_dir else []
    prefixes = []
    for i in range(len(parts), -1, -1):
        prefix = "/".join(parts[:i]).strip("/")
        prefixes.append(prefix)
    # Ensure uniqueness
    seen = set()
    ordered = []
    for p in prefixes:
        if p not in seen:
            ordered.append(p)
            seen.add(p)
    # Scan for first existing rule from most specific to least
    for p in ordered:
        if p in RULES and p != "":
            return RULES[p]
    # Finally, default
    return RULES.get("default", {"factor": 1.0})

# ------------------------------
# Bounds helpers (version-proof)
# ------------------------------
def object_world_bounds(o):
    """Return (bmin, bmax) in WORLD space, regardless of API version."""
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

# ------------------------------
# Export helpers (bpy-based)
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
    # Deselect all
    for ob in bpy.data.objects:
        ob.select_set(False)
    # Select targets
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

    # Compute size (pre-scale)
    bmin, bmax = combined_bounds(objs)
    size = bmax - bmin
    max_dim = float(np.max(size))
    if not np.isfinite(max_dim) or max_dim <= 0:
        print(f"    Non-positive size (max_dim={max_dim}), skipping.")
        return False

    # Determine uniform scale factor from rule
    if "factor" in rule:
        factor = float(rule["factor"])
    elif "target_size" in rule:
        factor = float(rule["target_size"]) / max_dim
    else:
        factor = 1.0

    # Apply scale
    if factor != 1.0:
        for o in objs:
            o.set_scale(o.get_scale() * factor)

    # Recenter to origin
    bmin, bmax = combined_bounds(objs)
    center = 0.5 * (bmin + bmax)
    for o in objs:
        o.set_location(o.get_location() - center)

    # Optional: place on floor
    if args.floor:
        bmin, bmax = combined_bounds(objs)
        dz = -float(bmin[2])
        if dz != 0.0:
            for o in objs:
                loc = o.get_location()
                o.set_location([loc[0], loc[1], loc[2] + dz])

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
# Walk and apply
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

        # Choose rule by most specific subfolder prefix
        rule = match_rule(rel_dir)

        out_dir_full = os.path.join(args.out_dir, rel_dir)
        out_path = os.path.join(out_dir_full, f"{name}_normalized{ext}")

        processed += 1
        if normalize_and_export(in_path, out_path, rule):
            saved += 1

print(f"Done. Processed {processed} file(s), saved {saved}.")
