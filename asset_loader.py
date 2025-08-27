# asset_loader.py
import os
import blenderproc as bproc
from typing import List

class AssetLoader:
    def __init__(self, asset_dir):
        self.asset_dir = asset_dir
        self.loaded_objs: List[List[bproc.types.MeshObject]] = []

    def load_assets(self):
        self.loaded_objs = []
        # Recursively find all .obj and .blend files
        files = []
        for root, _, filenames in os.walk(self.asset_dir):
            for f in filenames:
                if f.endswith((".obj", ".blend")):
                    files.append(os.path.join(root, f))
        for path in files:
            loaded = self._load_asset(path)
            loaded = [o for o in loaded if isinstance(o, bproc.types.MeshObject)]
            if loaded:
                self.loaded_objs.append(loaded)
        return self.loaded_objs

    def _load_asset(self, path):
        if path.endswith(".obj"):
            return bproc.loader.load_obj(path)
        elif path.endswith(".blend"):
            return bproc.loader.load_blend(path)
        return []

    def get_loaded_objs(self):
        return self.loaded_objs
