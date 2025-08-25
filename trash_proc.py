import blenderproc as bproc
import os, random
import numpy as np

# 1. Init BlenderProc
bproc.init()

# 2. Collect all assets (OBJ + BLEND) from folder
asset_dir = "/home/alex/projects/trash_meshes/bottle/beer_bottle"
files = [f for f in os.listdir(asset_dir) if f.endswith((".obj", ".blend"))]

loaded_objs = []

for f in files:
    path = os.path.join(asset_dir, f)
    if f.endswith(".obj"):
        objs = bproc.loader.load_obj(path, load_materials=True)
        loaded_objs.extend(objs)
    elif f.endswith(".blend"):
        objs = bproc.loader.load_blend(path)
        loaded_objs.extend(objs)

# 3. Randomly place each object
for obj in loaded_objs:
    obj.set_location([
        random.uniform(-1, 1),  # X
        random.uniform(-1, 1),  # Y
        random.uniform(0, 1)    # Z
    ])
    obj.set_rotation_euler([
        random.uniform(0, np.pi),  # X
        random.uniform(0, np.pi),  # Y
        random.uniform(0, np.pi)   # Z
    ])
    # Optional: random scale
    #scale = random.uniform(0.5, 1.5)
    #obj.set_scale([scale, scale, scale])

# 4. Compute scene bounding box (for camera placement)
mins, maxs = [], []
for o in loaded_objs:
    bb = o.get_bound_box()
    mins.append(bb.min(axis=0))
    maxs.append(bb.max(axis=0))

scene_min = np.min(np.vstack(mins), axis=0)
scene_max = np.max(np.vstack(maxs), axis=0)
center    = (scene_min + scene_max) / 2.0
extent    = scene_max - scene_min
base_radius = max(extent.max(), 1.0) * 2.5  # how far the camera sits

# 5. Add camera poses around scene
def sph_to_cart(radius, az_deg, el_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    x = radius * np.cos(el) * np.cos(az)
    y = radius * np.cos(el) * np.sin(az)
    z = radius * np.sin(el)
    return np.array([x, y, z])

for i in range(3):  # three random views
    radius = base_radius * np.random.uniform(0.9, 1.2)
    az = np.random.uniform(0, 360)
    el = np.random.uniform(10, 45)
    cam_pos = center + sph_to_cart(radius, az, el)

    forward = center - cam_pos
    cam_pose = bproc.math.build_transformation_mat(
        cam_pos.tolist(),
        bproc.camera.rotation_from_forward_vec(forward.tolist(),
                                               inplane_rot=np.random.uniform(0, 2*np.pi))
    )
    bproc.camera.add_camera_pose(cam_pose)


# 6. Add lights
light = bproc.types.Light()
light.set_type("SUN")
light.set_location([0, 0, 5])
light.set_energy(5)

# 7. Render and save
bproc.renderer.set_output_format("PNG")
bproc.renderer.set_max_amount_of_samples(64)   # new API
bproc.renderer.set_render_devices("CPU")  # or "GPU" if supported

images = bproc.renderer.render()
bproc.writer.write_hdf5("output/", images)
