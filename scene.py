import random
import numpy as np
import blenderproc as bproc
import itertools
from utility import sph_to_cart
import os
import glob
import math

class Scene:
    def __init__(self, all_loaded_groups):
        self.all_loaded_groups = all_loaded_groups

    def sample_pose(self, obj: bproc.types.MeshObject):
        obj.set_location(np.random.uniform([-5, -5, -5], [5, 5, 5]))
        obj.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi, np.pi, np.pi]))

    def place_objects_randomly(self):
        bproc.object.sample_poses(list(itertools.chain.from_iterable(self.all_loaded_groups)), sample_pose_func=self.sample_pose)

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

    def add_camera_in_room(self, min_height: float = 1.4, max_height: float = 1.7):
            """
            Place one camera somewhere inside the room (above the floor) and make it look at the room center.
            """
            if not hasattr(self, "room_objects"):
                raise RuntimeError("No room objects found; call add_random_room() first.")

            # Find a floor object
            floor_objs = [o for o in self.room_objects if "Floor" in o.get_name()]
            if not floor_objs:
                raise RuntimeError("No floor object found in the room; cannot place camera.")
            floor = floor_objs[0]

            # Sample a camera location above the floor
            cam_location = bproc.sampler.upper_region(
                objects_to_sample_on=[floor],
                min_height=min_height,
                max_height=max_height,
                use_ray_trace_check=True
            )

            # Aim at room center (from floor bbox)
            bb = np.array(floor.get_bound_box())
            room_center = bb.mean(axis=0)

            forward_vec = room_center - cam_location
            if np.linalg.norm(forward_vec) < 1e-6:
                # Edge case: sampled exactly at center; point along -Z
                forward_vec = np.array([0.0, 0.0, -1.0])

            R = bproc.camera.rotation_from_forward_vec(forward_vec)  # 3x3

            # Build a 4x4 cam2world matrix
            cam2world = np.eye(4, dtype=np.float64)
            cam2world[:3, :3] = R
            cam2world[:3, 3]  = cam_location

            # Register the pose
            bproc.camera.add_camera_pose(cam2world)

    def add_random_background(self, bg_folder, strength=1.0):
        """
        Pick a random image (jpg/png/hdr/exr) from bg_folder and set it as the world background.
        Returns the selected path.
        """
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.hdr", "*.exr"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(bg_folder, p)))
        if not files:
            raise RuntimeError(f"No background images found in: {bg_folder}")
        chosen = random.choice(files)

        bproc.world.set_world_background_hdr_img(chosen, strength=strength)
        
        return chosen

    def add_random_room(self, cc_material_dir, pix3d_dir, amount=50,
                        target_longest_side_range=(1.0, 1.1),
                        used_floor_area=9.0,
                        wall_height=2.7):
        """
        Build a random room and populate it with a random subset of Pix3D meshes.

        Args:
            cc_material_dir: path to CCTextures/ambientCG materials.
            pix3d_dir: root folder of Pix3D dataset.
            amount: how many furniture objects to sample.
            target_longest_side_range: clamp each object's longest bbox side to this [min,max] (meters).
            used_floor_area: floor area in m² for the room.
            wall_height: room wall height in meters.
        """

        materials = bproc.loader.load_ccmaterials(cc_material_dir)

        # Find OBJ files under Pix3D
        patterns = [
            os.path.join(pix3d_dir, "*", "*.obj"),
            os.path.join(pix3d_dir, "*", "*", "*.obj"),
            os.path.join(pix3d_dir, "*", "*", "model.obj"),
        ]
        candidates = sorted({p for pat in patterns for p in glob.glob(pat)})
        if not candidates:
            raise RuntimeError(f"No OBJ files found under Pix3D dir: {pix3d_dir}")

        chosen = random.sample(candidates, k=min(amount, len(candidates)))

        def normalize_scale(mesh_obj, tgt_range):
            """Scale so the longest bbox side is within tgt_range (meters)."""
            bb = np.asarray(mesh_obj.get_bound_box())  # (8,3)
            ext = bb.max(axis=0) - bb.min(axis=0)
            longest = float(ext.max())
            if longest <= 1e-6:
                return  # skip degenerate
            # Pick a random target size within range to add variation
            target = random.uniform(*tgt_range)
            s = target / longest
            mesh_obj.set_scale([s, s, s])

        interior_objects = []
        for obj_path in chosen:
            loaded = bproc.loader.load_obj(obj_path)
            for o in loaded:
                # Normalize to sensible furniture size instead of 100x
                normalize_scale(o, target_longest_side_range)
                
                # fix the rotation bug fixed by object import
                o.persist_transformation_into_mesh(location=False, rotation=True, scale=False)

                # remove offset
                o.move_origin_to_bottom_mean_point()

                o.set_cp("dataset", "pix3d")
            interior_objects.extend(loaded)

        # Let the constructor build the room AND place interior_objects inside it
        room_objects = bproc.constructor.construct_random_room(
            used_floor_area=used_floor_area,
            interior_objects=interior_objects,
            materials=materials,
            amount_of_extrusions=3,
            corridor_width=1.2,
            wall_height=wall_height,
            only_use_big_edges=False,
            amount_of_objects_per_sq_meter=1.0
        )

        # Optional: make the ceiling softly emissive
        bproc.lighting.light_surface(
            [o for o in room_objects if "Ceiling" in o.get_name()],
            emission_strength=random.uniform(0.5, 1.0)
        )

        self.room_objects = room_objects  # these are shell objects, not furniture
        return room_objects
    

    def place_objects_in_room(self, scale: float = 0.08):
        if not hasattr(self, "room_objects"):
            raise RuntimeError("No room objects found; call add_random_room() first.")
        
        floor_objs = [o for o in self.room_objects if "Floor" in o.get_name()]
        if not floor_objs:
            raise RuntimeError("No floor object found in the room; cannot place objects.")

        # scale all objects in all_loaded_groups by a single uniform factor (argument)
        flat_objs = list(itertools.chain.from_iterable(self.all_loaded_groups))
        if flat_objs and abs(scale - 1.0) > 1e-6:
            for o in flat_objs:
                s = np.array(o.get_scale(), dtype=float)
                o.set_scale((s * float(scale)).tolist())

        # Define a sampling function that closes over floor_objs
        def sample_pose_surface(obj: bproc.types.MeshObject):
            obj.set_location(bproc.sampler.upper_region(
                objects_to_sample_on=floor_objs,
                min_height=1,
                max_height=4,
                use_ray_trace_check=False
            ))
            obj.set_rotation_euler(
                np.random.uniform([0, 0, 0], [np.pi * 2, np.pi * 2, np.pi * 2])
            )

        bproc.object.sample_poses_on_surface(
            list(itertools.chain.from_iterable(self.all_loaded_groups)),
            floor_objs[0],
            max_distance=10,
            min_distance=0.00001,
            max_tries=500,
            sample_pose_func=sample_pose_surface
        )


    def add_light(self, light_type="SUN", location=[0,0,5], energy=10):
        light = bproc.types.Light()
        light.set_type(light_type)
        light.set_location(location)
        light.set_energy(energy)