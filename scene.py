import random
import numpy as np
import blenderproc as bproc
import itertools
from utility import sph_to_cart
import os
import glob

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
    
    
    def add_random_room(self, cc_material_dir, pix3d_dir, amount=15, scale_range=(100, 150)):
        """
        Build a random room (walls/floor/ceiling) and populate it with a random subset
        of Pix3D meshes instead of the (now-unavailable) IKEA dataset.

        Args:
            cc_material_dir: path to CCTextures/ambientCG materials (for walls/floor/ceiling)
            pix3d_dir: root folder of Pix3D dataset
            amount: how many furniture objects to sample for this room
            scale_range: random uniform scale applied to each loaded object

        Returns:
            room_objects: list of MeshObjects that form the room shell
        """
        # 1) Materials for the room shell
        materials = bproc.loader.load_ccmaterials(cc_material_dir)

        # 2) Gather Pix3D OBJ paths (handles typical Pix3D layouts)
        #    Many Pix3D releases store meshes under category/model/*.obj or category/*/*.obj
        candidates = set()
        patterns = [
            os.path.join(pix3d_dir, "*", "*.obj"),
            os.path.join(pix3d_dir, "*", "*", "*.obj"),
            os.path.join(pix3d_dir, "*", "*", "model.obj"),   # common Pix3D naming
        ]
        for pat in patterns:
            for p in glob.glob(pat):
                # Heuristic: avoid accidental non-mesh objs (if any)
                name = os.path.basename(p).lower()
                if name.endswith(".obj"):
                    candidates.add(p)

        candidates = sorted(candidates)
        if not candidates:
            raise RuntimeError(f"No OBJ files found under Pix3D dir: {pix3d_dir}")

        # 3) Sample a subset to keep memory/render times reasonable
        chosen = random.sample(candidates, k=min(amount, len(candidates)))

        # 4) Load chosen meshes (with materials if MTL/textures are present)
        interior_objects = []
        for obj_path in chosen:
            loaded = bproc.loader.load_obj(obj_path)
            for o in loaded:
                # Random gentle rescale (Pix3D units vary a bit)
                s = random.uniform(*scale_range)
                o.set_scale([s, s, s])
                o.set_cp("dataset", "pix3d")
            interior_objects.extend(loaded)

        # 5) Construct a random room and let BlenderProc place the objects in it
        room_objects = bproc.constructor.construct_random_room(
            interior_objects=interior_objects,
            materials=materials,
            used_floor_area=500,
            amount_of_extrusions=3,
            corridor_width=1.5
        )

        # Optional: make the ceiling a soft area light
        bproc.lighting.light_surface(
            [o for o in room_objects if "Ceiling" in o.get_name()],
            emission_strength=2.0
        )

        self.room_objects = room_objects
        return room_objects
    

    def place_objects_in_room(self):
        if not hasattr(self, "room_objects"):
            raise RuntimeError("No room objects found; call add_random_room() first.")
        
        floor_objs = [o for o in self.room_objects if "Floor" in o.get_name()]
        if not floor_objs:
            raise RuntimeError("No floor object found in the room; cannot place objects.")

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
            max_distance=0.1,
            min_distance=0.05,
            max_tries=500,
            sample_pose_func=sample_pose_surface  # <- not self.sample_pose_surface
        )


    def add_light(self, light_type="SUN", location=[0,0,5], energy=10):
        light = bproc.types.Light()
        light.set_type(light_type)
        light.set_location(location)
        light.set_energy(energy)