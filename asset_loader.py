# asset_loader.py
import os
import blenderproc as bproc
from typing import List

class AssetLoader:
    def __init__(self, asset_dir):
        self.asset_dir = asset_dir
        self.loaded_objs: List[List[bproc.types.MeshObject]] = []

    def load_assets(self):
        files = [f for f in os.listdir(self.asset_dir) if f.endswith((".obj", ".blend"))]
        self.loaded_objs = []
        for f in files:
            path = os.path.join(self.asset_dir, f)
            loaded = self._load_asset(path)
            # keep only MeshObjects
            #loaded = [o for o in loaded if isinstance(o, bproc.types.MeshObject)]
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
