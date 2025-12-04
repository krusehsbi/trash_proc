import blenderproc as bproc
import os
import random
import math
import mathutils
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


from asset_loader import AssetLoader

# 1) Init BlenderProc
bproc.init()

# 2) Load the .blend file, including mesh, camera and light objects
scene_path = "conveyor.blend"
objs = bproc.loader.load_blend(scene_path, obj_types=["mesh", "camera", "light"])

# 3) Pick the camera object from the loaded objects
#    - if you know your camera's name, put it here (e.g. "Camera" or "Camera.001")
cam_obj = bproc.filter.one_by_attr(objs, "name", "camera")

# 4) Build a cam2world matrix from that camera's location + rotation
cam_location = cam_obj.get_location()
cam_rotation = cam_obj.get_rotation()  # Euler angles
cam2world = bproc.math.build_transformation_mat(cam_location, cam_rotation)

# 5) Register this as the only camera pose
bproc.camera.add_camera_pose(cam2world)

# (Optional) Set resolution (intrinsics come from Blender defaults / FOV)
bproc.camera.set_resolution(1920, 1080)

# --- New: load assets and place them randomly on the conveyor ---
# Find conveyor mesh by name
conveyor = bproc.filter.one_by_attr(objs, "name", "conveyor")
if conveyor is None:
    raise RuntimeError("Could not find an object named 'conveyor' in the scene.")
conveyor.enable_rigidbody(active=False)

# instantiate AssetLoader and load assets
asset_dir = "/home/alex/projects/trash_proc/scaled_assets"
loader = AssetLoader(asset_dir=asset_dir)
loaded_groups = loader.load_assets(asset_dir=asset_dir, clear=True, group_parts_as_one=True)


# Apply normalization and basic fixes (persist rotation bug, move origin to bottom)
for group in loaded_groups:
    if not group:
        continue
    mesh_obj = group[0]
    try:
        # fix import rotation issues if present (scene.py did rotation fix)
        mesh_obj.persist_transformation_into_mesh(location=True, rotation=True, scale=True)
    except Exception:
        pass
    try:
        mesh_obj.move_origin_to_bottom_mean_point()
    except Exception:
        pass

# Place each loaded asset at a random position in WORLD coordinates
for group in loaded_groups:
    if not group:
        continue
    mesh_obj = group[0]

    yaw = random.uniform(0.25 * math.pi, 2.0 * math.pi)

    # set location and rotation (Euler XYZ)
    try:
        loc = bproc.sampler.upper_region(
                objects_to_sample_on=[conveyor],
                min_height=0.05, max_height=0.15,
                use_ray_trace_check=True
            )
        mesh_obj.set_location(loc)
        mesh_obj.set_rotation_euler([yaw, yaw, yaw])
    except Exception as e:
        print(e)
        print("Failed to set location/rotation for an object.")

    mesh_obj.enable_rigidbody(active=True)

bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=2, max_simulation_time=4, check_object_interval=1)


# 6) Render one frame from that pose
data = bproc.renderer.render()

# 7) Write output (RGB will be in data["colors"], saved to HDF5)
out_dir = "/home/alex/projects/trash_proc/output_conveyor"
os.makedirs(out_dir, exist_ok=True)
bproc.writer.write_hdf5(out_dir, data)
