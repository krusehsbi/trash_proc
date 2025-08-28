import blenderproc as bproc
import os, random
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from asset_loader import AssetLoader
from scene import Scene
import json

# 1. Init BlenderProc
bproc.init()

# 2. Collect all assets (OBJ + BLEND) from folder
loader = AssetLoader()  # reuse one loader
with open(ROOT / "configs/class_mapping.json", "r") as f:
    class_mappings = json.load(f)

for category in class_mappings:
    category_id = category["class_id"]
    class_dir = category["class_dir"]
    name = category["class_name"]
    category_dir = os.path.join(ROOT, "assets", class_dir)
    if not os.path.exists(category_dir):
        print(f"[warn] Category directory does not exist: {category_dir}")
        continue

    # append results into loader.all_loaded_groups (default behaviour)
    loader.load_assets(asset_dir=category_dir, category_id=category_id, category_name=name)

#3. Randomly place objects in scene
# use accumulated groups:
all_loaded_groups = loader.get_all_loaded_groups()
print("Loaded object groups:", all_loaded_groups)

scene = Scene(all_loaded_groups)
scene.place_objects_randomly()

# 4. Compute camera radius from scene (for camera placement)
center, base_radius = scene.find_camera_radius(distance_factor=1.5)

# 5. Add camera poses around scene
for i in range(3):  # three random views
    scene.add_camera_poses(center, base_radius)


# 6. Add lights
light = bproc.types.Light()
light.set_type("SUN")
light.set_location([0, 0, 5])
light.set_energy(10)

# 7. Render and save
bproc.renderer.set_output_format("PNG")
bproc.renderer.set_max_amount_of_samples(128)   # new API
bproc.renderer.set_render_devices("CPU")  # or "GPU" if supported

images = bproc.renderer.render()
bproc.writer.write_hdf5("output/", images)

# 8. Save COCO annotations
seg_data = bproc.renderer.render_segmap(map_by=["class", "instance"])
bproc.writer.write_coco_annotations(
    output_dir="output/coco_data",
    instance_segmaps=seg_data["instance_segmaps"],
    instance_attribute_maps=seg_data["instance_attribute_maps"],
    colors=images["colors"],
    color_file_format="JPEG"
)


