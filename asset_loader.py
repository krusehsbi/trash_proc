# asset_loader.py
import os
import blenderproc as bproc
from typing import List, Optional

class AssetLoader:
    def __init__(self, asset_dir: Optional[str] = None):
        """
        If asset_dir provided here it will be used as default.
        load_assets can be called repeatedly for different folders;
        by default results are appended to all_loaded_groups.
        """
        self.asset_dir = asset_dir
        self.loaded_objs: List[List[bproc.types.MeshObject]] = []       # last load
        self.all_loaded_groups: List[List[bproc.types.MeshObject]] = []  # accumulated across calls

    def _iter_asset_files(self, asset_dir: str):
        for root, _, filenames in os.walk(asset_dir):
            for f in filenames:
                if f.lower().endswith((".obj", ".blend")):
                    yield os.path.join(root, f)

    def load_assets(
        self,
        asset_dir: Optional[str] = None,
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        assign_cp: bool = True,
        clear: bool = False,
        group_parts_as_one: bool = True,
    ) -> List[List[bproc.types.MeshObject]]:
        """
        Load assets from asset_dir (or default). Returns list-of-lists where each sublist
        contains MeshObjects loaded from one file.
        If assign_cp=True and category_id/name provided, they are set on each loaded object.
        By default results are appended to internal all_loaded_groups; pass clear=True to reset.
        """
        asset_dir = asset_dir or self.asset_dir
        if not asset_dir:
            raise ValueError("asset_dir must be provided either in constructor or to load_assets()")

        if clear:
            self.all_loaded_groups = []

        self.loaded_objs = []
        for path in sorted(self._iter_asset_files(asset_dir)):
            loaded = self._load_asset(path)
            mesh_objs = [o for o in loaded if isinstance(o, bproc.types.MeshObject)]
            if not mesh_objs:
                continue
            
            if assign_cp and (category_id is not None or category_name is not None):
                for o in mesh_objs:
                    if category_id is not None:
                        o.set_cp("category_id", category_id)
                    if category_name is not None:
                        o.set_cp("category_name", category_name)
                    
            if group_parts_as_one:
                mesh_objs[0].join_with_other_objects(mesh_objs[1:])
                self.loaded_objs.append([mesh_objs[0]])
                self.all_loaded_groups.append([mesh_objs[0]])
            else:
                self.loaded_objs.append(mesh_objs)
                self.all_loaded_groups.append(mesh_objs)

        return self.loaded_objs

    def _load_asset(self, path: str):
        if path.lower().endswith(".obj"):
            return bproc.loader.load_obj(path)
        elif path.lower().endswith(".blend"):
            return bproc.loader.load_blend(path)
        return []

    def get_loaded_objs(self) -> List[List[bproc.types.MeshObject]]:
        return self.loaded_objs

    def get_all_loaded_groups(self) -> List[List[bproc.types.MeshObject]]:
        return self.all_loaded_groups
