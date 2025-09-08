import random, math
import bpy
import blenderproc as bproc

class Weathering:
    """Category-agnostic 'trash-ify': subtle deforms + material aging."""
    def __init__(
        self,
        *,
        # probabilities (apply independently)
        p_displace=0.65,
        p_simple=0.45,
        p_lattice=0.25,
        p_axis_scale=0.60,
        # intensities (good for ~0.1–1 m objects)
        disp_noise_scale=(0.2, 1.0),
        disp_strength_per_diag=(0.003, 0.02),
        simple_angle_deg=(5.0, 15.0),      # for BEND/TWIST
        simple_factor=(-0.10, 0.10),       # for TAPER/STRETCH
        lattice_jitter_per_diag=0.03,
        axis_scale_delta=(-0.07, 0.05),
        # material aging
        dust_strength=(0.12, 0.28),
        dust_scale=(0.02, 0.08),
        roughness_jitter=(-0.20, 0.20),
        basecolor_mult=(0.85, 0.95),
        # general
        apply_modifiers=False,
        min_diag=0.05,
        max_diag=None,
        seed=None,
    ):
        self.p_displace = p_displace
        self.p_simple = p_simple
        self.p_lattice = p_lattice
        self.p_axis_scale = p_axis_scale

        self.disp_noise_scale = disp_noise_scale
        self.disp_strength_per_diag = disp_strength_per_diag
        self.simple_angle_deg = simple_angle_deg
        self.simple_factor = simple_factor
        self.lattice_jitter_per_diag = lattice_jitter_per_diag
        self.axis_scale_delta = axis_scale_delta

        self.dust_strength = dust_strength
        self.dust_scale = dust_scale
        self.roughness_jitter = roughness_jitter
        self.basecolor_mult = basecolor_mult

        self.apply_modifiers = apply_modifiers
        self.min_diag = min_diag
        self.max_diag = max_diag
        if seed is not None:
            random.seed(seed)

    # -------- public API --------
    def apply_to_groups(self, groups):
        for group in groups:
            for bp_obj in group:
                self._process(bp_obj)

    def apply_to_all_scene_meshes(self):
        for bp_obj in bproc.scene.get_objects():
            self._process(bp_obj)

    # -------- internals --------
    def _process(self, bp_obj):
        bpy_obj = bp_obj.blender_obj
        if bpy_obj is None or bpy_obj.type != "MESH":
            return

        diag = self._diag(bpy_obj)
        if diag < self.min_diag: return
        if self.max_diag is not None and diag > self.max_diag: return

        # geometry (probabilistic)
        if random.random() < self.p_displace: self._add_displace(bpy_obj, diag)
        if random.random() < self.p_simple:   self._add_simple(bpy_obj, diag)
        if random.random() < self.p_lattice:  self._add_lattice(bpy_obj, diag)
        if random.random() < self.p_axis_scale: self._axis_scale(bpy_obj)

        # materials
        self._age_materials(bpy_obj)

    # -- geometry ops --
    def _add_displace(self, bpy_obj, diag):
        tex = bpy.data.textures.new("wx_disp_tex", type="CLOUDS")
        tex.noise_scale = random.uniform(*self.disp_noise_scale)
        m = bpy_obj.modifiers.new(name="wx_displace", type="DISPLACE")
        m.texture = tex
        kmin, kmax = self.disp_strength_per_diag
        m.strength = random.uniform(kmin, kmax) * diag
        m.mid_level = 0.5
        self._maybe_apply(bpy_obj, m)

    def _add_simple(self, bpy_obj, diag):
        m = bpy_obj.modifiers.new(name="wx_simple", type="SIMPLE_DEFORM")
        m.deform_method = random.choice(["BEND", "TWIST", "TAPER", "STRETCH"])
        m.deform_axis = random.choice(["X", "Y", "Z"])
        size = min(1.0, max(0.3, diag))
        if m.deform_method in {"BEND", "TWIST"}:
            amin, amax = self.simple_angle_deg
            m.angle = math.radians(random.uniform(amin, amax) * size)
        else:
            fmin, fmax = self.simple_factor
            m.factor = random.uniform(fmin, fmax) * size
        self._maybe_apply(bpy_obj, m)

    def _add_lattice(self, bpy_obj, diag):
        lat_data = bpy.data.lattices.new("wx_lat_data")
        lat = bpy.data.objects.new("wx_lat", lat_data)
        bpy.context.scene.collection.objects.link(lat)
        lat_data.points_u = random.choice([2,3,4])
        lat_data.points_v = random.choice([2,3,4])
        lat_data.points_w = random.choice([2,3,4])

        lat.location = bpy_obj.location
        lat.rotation_euler = bpy_obj.rotation_euler
        lat.scale = bpy_obj.dimensions * 0.6

        m = bpy_obj.modifiers.new(name="wx_lattice_mod", type="LATTICE")
        m.object = lat

        j = self.lattice_jitter_per_diag * diag
        for p in lat_data.points:
            p.co_deform[0] += random.uniform(-j, j)
            p.co_deform[1] += random.uniform(-j, j)
            p.co_deform[2] += random.uniform(-j, j)

        self._maybe_apply(bpy_obj, m)
        if self.apply_modifiers:
            try: bpy.data.objects.remove(lat, do_unlink=True)
            except Exception: pass

    def _axis_scale(self, bpy_obj):
        rmin, rmax = self.axis_scale_delta
        sx = 1 + random.uniform(rmin, rmax)
        sy = 1 + random.uniform(rmin, rmax)
        sz = 1 + random.uniform(rmin, rmax)
        bpy_obj.scale[0] *= sx; bpy_obj.scale[1] *= sy; bpy_obj.scale[2] *= sz

    # -- material ops --
    def _age_materials(self, bpy_obj):
        mats = getattr(bpy_obj.data, "materials", None) or []
        for bpy_mat in mats:
            if bpy_mat is None or not getattr(bpy_mat, "use_nodes", False):
                continue
            try:
                bp_mat = bproc.types.Material(bpy_mat)  # safe now (nodes only)
            except Exception:
                continue

            nt = getattr(bpy_mat, "node_tree", None)
            if nt:
                principled = next((n for n in nt.nodes if getattr(n, "type", "") == "BSDF_PRINCIPLED"), None)
                if principled:
                    # roughness jitter (multiplicative, clamped)
                    r_in = principled.inputs.get("Roughness")
                    if r_in and not r_in.is_linked:
                        cur = float(r_in.default_value)
                        mult = 1 + random.uniform(*self.roughness_jitter)
                        r_in.default_value = max(0.0, min(1.0, cur * mult))
                    # base color multiplier
                    c_in = principled.inputs.get("Base Color")
                    if c_in and not c_in.is_linked:
                        r,g,b,a = c_in.default_value
                        m = random.uniform(*self.basecolor_mult)
                        c_in.default_value = (r*m, g*m, b*m, a)

            # dust (skip failures silently)
            try:
                bproc.material.add_dust(
                    bp_mat,
                    strength=random.uniform(*self.dust_strength),
                    texture_scale=random.uniform(*self.dust_scale),
                )
            except Exception:
                pass

    # -- utils --
    def _diag(self, bpy_obj):
        d = bpy_obj.dimensions
        return float((d.x*d.x + d.y*d.y + d.z*d.z) ** 0.5)

    def _maybe_apply(self, bpy_obj, mod):
        if not self.apply_modifiers: return
        try:
            bpy.context.view_layer.objects.active = bpy_obj
            bpy.ops.object.modifier_apply(modifier=mod.name)
        except Exception:
            pass
# --- end weathering.py ---
# --- weathering.py ---
import random, math
import bpy
import blenderproc as bproc

class Weathering:
    """Category-agnostic 'trash-ify': subtle deforms + material aging."""
    def __init__(
        self,
        *,
        # probabilities (apply independently)
        p_displace=0.65,
        p_simple=0.45,
        p_lattice=0.25,
        p_axis_scale=0.60,
        # intensities (good for ~0.1–1 m objects)
        disp_noise_scale=(0.2, 1.0),
        disp_strength_per_diag=(0.003, 0.02),
        simple_angle_deg=(5.0, 15.0),      # for BEND/TWIST
        simple_factor=(-0.10, 0.10),       # for TAPER/STRETCH
        lattice_jitter_per_diag=0.03,
        axis_scale_delta=(-0.07, 0.05),
        # material aging
        dust_strength=(0.12, 0.28),
        dust_scale=(0.02, 0.08),
        roughness_jitter=(-0.20, 0.20),
        basecolor_mult=(0.85, 0.95),
        # general
        apply_modifiers=False,
        min_diag=0.05,
        max_diag=None,
        seed=None,
    ):
        self.p_displace = p_displace
        self.p_simple = p_simple
        self.p_lattice = p_lattice
        self.p_axis_scale = p_axis_scale

        self.disp_noise_scale = disp_noise_scale
        self.disp_strength_per_diag = disp_strength_per_diag
        self.simple_angle_deg = simple_angle_deg
        self.simple_factor = simple_factor
        self.lattice_jitter_per_diag = lattice_jitter_per_diag
        self.axis_scale_delta = axis_scale_delta

        self.dust_strength = dust_strength
        self.dust_scale = dust_scale
        self.roughness_jitter = roughness_jitter
        self.basecolor_mult = basecolor_mult

        self.apply_modifiers = apply_modifiers
        self.min_diag = min_diag
        self.max_diag = max_diag
        if seed is not None:
            random.seed(seed)

    # -------- public API --------
    def apply_to_groups(self, groups):
        for group in groups:
            for bp_obj in group:
                self._process(bp_obj)

    def apply_to_all_scene_meshes(self):
        for bp_obj in bproc.scene.get_objects():
            self._process(bp_obj)

    # -------- internals --------
    def _process(self, bp_obj):
        bpy_obj = bp_obj.blender_obj
        if bpy_obj is None or bpy_obj.type != "MESH":
            return

        diag = self._diag(bpy_obj)
        if diag < self.min_diag: return
        if self.max_diag is not None and diag > self.max_diag: return

        # geometry (probabilistic)
        if random.random() < self.p_displace: self._add_displace(bpy_obj, diag)
        if random.random() < self.p_simple:   self._add_simple(bpy_obj, diag)
        if random.random() < self.p_lattice:  self._add_lattice(bpy_obj, diag)
        if random.random() < self.p_axis_scale: self._axis_scale(bpy_obj)

        # materials
        self._age_materials(bpy_obj)

    # -- geometry ops --
    def _add_displace(self, bpy_obj, diag):
        tex = bpy.data.textures.new("wx_disp_tex", type="CLOUDS")
        tex.noise_scale = random.uniform(*self.disp_noise_scale)
        m = bpy_obj.modifiers.new(name="wx_displace", type="DISPLACE")
        m.texture = tex
        kmin, kmax = self.disp_strength_per_diag
        m.strength = random.uniform(kmin, kmax) * diag
        m.mid_level = 0.5
        self._maybe_apply(bpy_obj, m)

    def _add_simple(self, bpy_obj, diag):
        m = bpy_obj.modifiers.new(name="wx_simple", type="SIMPLE_DEFORM")
        m.deform_method = random.choice(["BEND", "TWIST", "TAPER", "STRETCH"])
        m.deform_axis = random.choice(["X", "Y", "Z"])
        size = min(1.0, max(0.3, diag))
        if m.deform_method in {"BEND", "TWIST"}:
            amin, amax = self.simple_angle_deg
            m.angle = math.radians(random.uniform(amin, amax) * size)
        else:
            fmin, fmax = self.simple_factor
            m.factor = random.uniform(fmin, fmax) * size
        self._maybe_apply(bpy_obj, m)

    def _add_lattice(self, bpy_obj, diag):
        lat_data = bpy.data.lattices.new("wx_lat_data")
        lat = bpy.data.objects.new("wx_lat", lat_data)
        bpy.context.scene.collection.objects.link(lat)
        lat_data.points_u = random.choice([2,3,4])
        lat_data.points_v = random.choice([2,3,4])
        lat_data.points_w = random.choice([2,3,4])

        lat.location = bpy_obj.location
        lat.rotation_euler = bpy_obj.rotation_euler
        lat.scale = bpy_obj.dimensions * 0.6

        m = bpy_obj.modifiers.new(name="wx_lattice_mod", type="LATTICE")
        m.object = lat

        j = self.lattice_jitter_per_diag * diag
        for p in lat_data.points:
            p.co_deform[0] += random.uniform(-j, j)
            p.co_deform[1] += random.uniform(-j, j)
            p.co_deform[2] += random.uniform(-j, j)

        self._maybe_apply(bpy_obj, m)
        if self.apply_modifiers:
            try: bpy.data.objects.remove(lat, do_unlink=True)
            except Exception: pass

    def _axis_scale(self, bpy_obj):
        rmin, rmax = self.axis_scale_delta
        sx = 1 + random.uniform(rmin, rmax)
        sy = 1 + random.uniform(rmin, rmax)
        sz = 1 + random.uniform(rmin, rmax)
        bpy_obj.scale[0] *= sx; bpy_obj.scale[1] *= sy; bpy_obj.scale[2] *= sz

    # -- material ops --
    def _age_materials(self, bpy_obj):
        mats = getattr(bpy_obj.data, "materials", None) or []
        for bpy_mat in mats:
            if bpy_mat is None or not getattr(bpy_mat, "use_nodes", False):
                continue
            try:
                bp_mat = bproc.types.Material(bpy_mat)  # safe now (nodes only)
            except Exception:
                continue

            nt = getattr(bpy_mat, "node_tree", None)
            if nt:
                principled = next((n for n in nt.nodes if getattr(n, "type", "") == "BSDF_PRINCIPLED"), None)
                if principled:
                    # roughness jitter (multiplicative, clamped)
                    r_in = principled.inputs.get("Roughness")
                    if r_in and not r_in.is_linked:
                        cur = float(r_in.default_value)
                        mult = 1 + random.uniform(*self.roughness_jitter)
                        r_in.default_value = max(0.0, min(1.0, cur * mult))
                    # base color multiplier
                    c_in = principled.inputs.get("Base Color")
                    if c_in and not c_in.is_linked:
                        r,g,b,a = c_in.default_value
                        m = random.uniform(*self.basecolor_mult)
                        c_in.default_value = (r*m, g*m, b*m, a)

            # dust (skip failures silently)
            try:
                bproc.material.add_dust(
                    bp_mat,
                    strength=random.uniform(*self.dust_strength),
                    texture_scale=random.uniform(*self.dust_scale),
                )
            except Exception:
                pass

    # -- utils --
    def _diag(self, bpy_obj):
        d = bpy_obj.dimensions
        return float((d.x*d.x + d.y*d.y + d.z*d.z) ** 0.5)

    def _maybe_apply(self, bpy_obj, mod):
        if not self.apply_modifiers: return
        try:
            bpy.context.view_layer.objects.active = bpy_obj
            bpy.ops.object.modifier_apply(modifier=mod.name)
        except Exception:
            pass