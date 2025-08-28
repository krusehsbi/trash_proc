import random
import numpy as np
import blenderproc as bproc
from utility import sph_to_cart

class Scene:
    def __init__(self, all_loaded_groups):
        self.all_loaded_groups = all_loaded_groups

    # TODO: improve placement logic (e.g. avoid collisions)
    def place_objects_randomly(self):
        for obj_group in self.all_loaded_groups:
            # Generate one random transformation per group
            location = [
                random.uniform(-5, 5),  # X
                random.uniform(-5, 5),  # Y
                random.uniform(0, 5)    # Z
            ]
            rotation = [
                random.uniform(0, np.pi),  # X
                random.uniform(0, np.pi),  # Y
                random.uniform(0, np.pi)   # Z
            ]
            # Optional: random scale
            #scale = random.uniform(0.5, 1.5)
            for obj in obj_group:
                obj.set_location(location)
                obj.set_rotation_euler(rotation)
                #obj.set_scale([scale, scale, scale])

    def find_camera_radius(self, distance_factor=1.5):
        mins, maxs = [], []
        for obj_group in self.all_loaded_groups:
            for o in obj_group:
                if not hasattr(o, "get_bound_box"):
                    print(f"[warn] Skipping non-mesh object: {o}")
                    continue
                bb = o.get_bound_box()     # 8x3 array-like
                bb = np.asarray(bb)
                mins.append(bb.min(axis=0))
                maxs.append(bb.max(axis=0))

        if not mins:
            raise RuntimeError("No mesh objects with bounding boxes were loaded; check your asset folder and loader.")
        scene_min = np.min(np.vstack(mins), axis=0)
        scene_max = np.max(np.vstack(maxs), axis=0)
        center    = (scene_min + scene_max) / 2.0
        extent    = scene_max - scene_min
        base_radius = max(extent.max(), 1.0) * distance_factor # how far the camera sits
        return center, base_radius
    
    def add_camera_poses(self, center, base_radius):

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
