from enum import Enum

class TextureWrap(Enum):
    CLAMP_TO_EDGE = 33071
    MIRRORED_REPEAT = 33648
    REPEAT = 10497

class AlphaMode(Enum):
    BLEND = 'BLEND'
    MASK = 'MASK'
    OPAQUE = 'OPAQUE'

class Texture(object):
    def __init__(self, texture_entry, gltf_loader):
        index = 0
        self._name = texture_entry['name'] if ('name' in texture_entry) else 'texture_{}'.format(index)
        self._image = gltf_loader.get_images()[texture_entry['index']]
        self._index = texture_entry['index'] if ('index' in texture_entry) else 0
        self._texcoord_index = texture_entry['texCoord'] if ('texCoord' in texture_entry) else 0
        self._gltf_image = gltf_loader.get_images()[self._index]
        sampler = gltf_loader.json_data['samplers'][texture_entry['sampler']] if ('sampler' in texture_entry) else (gltf_loader.json_data['samplers'][0] if ('samplers' in gltf_loader.json_data) else {})
        self._wrap_s = TextureWrap(sampler['wrapS']) if ('wrapS' in sampler) else TextureWrap.REPEAT
        self._wrap_t = TextureWrap(sampler['wrapT']) if ('wrapT' in sampler) else TextureWrap.REPEAT
        self._extensions = {}
        self._tt_offset = [0.0,0.0]
        self._tt_scale = [1.0,1.0]
        self._tt_rotation = 0.0

        if 'extensions' in texture_entry and 'KHR_texture_transform' in texture_entry['extensions']:
            self._extensions['KHR_texture_transform'] = KHRTextureTransform(texture_entry['extensions']['KHR_texture_transform'])
            self._tt_offset = self._extensions['KHR_texture_transform']._offset
            self._tt_scale = self._extensions['KHR_texture_transform']._scale
            self._tt_rotation = self._extensions['KHR_texture_transform']._rotation

    def get_name(self):
        return self._name

    def get_image_path(self):
        return self._image.get_image_path()

    def get_wrap_s(self):
        return self._wrap_s

    def get_wrap_t(self):
        return self._wrap_t

    def write_to_directory(self, output_directory, channels, texture_prefix=""):
        return self._image.write_to_directory(output_directory, channels, texture_prefix, self._tt_offset, self._tt_scale, self._tt_rotation)

    def get_texcoord_index(self):
        return self._texcoord_index

    @property
    def extensions(self):
        return self._extensions

class NormalTexture(Texture):
    def __init__(self, normal_texture_entry, gltf_loader):
        super(NormalTexture, self).__init__(normal_texture_entry, gltf_loader)
        self._scale = normal_texture_entry['scale'] if ('scale' in normal_texture_entry) else 1.0      

    @property
    def scale(self):
        return self._scale


class OcclusionTexture(Texture):
    def __init__(self, occlusion_texture_entry, gltf_loader):
        super(OcclusionTexture, self).__init__(occlusion_texture_entry, gltf_loader)
        self._strength = occlusion_texture_entry['strength'] if ('strength' in occlusion_texture_entry) else 1.0

    @property
    def strength(self):
        return self._strength

class PbrMetallicRoughness:
    def __init__(self, pbr_metallic_roughness_entry, gltf_loader):
        self._name = pbr_metallic_roughness_entry['name'] if ('name' in pbr_metallic_roughness_entry) else 'pbr_mat_roughness_texture'
        self._base_color_factor = pbr_metallic_roughness_entry['baseColorFactor'] if ('baseColorFactor' in pbr_metallic_roughness_entry) else [1.0,1.0,1.0, 1.0]
        self._metallic_factor = pbr_metallic_roughness_entry['metallicFactor'] if ('metallicFactor' in pbr_metallic_roughness_entry) else 1.0
        self._roughness_factor = pbr_metallic_roughness_entry['roughnessFactor'] if ('roughnessFactor' in pbr_metallic_roughness_entry) else 1.0
        self._base_color_texture = Texture(pbr_metallic_roughness_entry['baseColorTexture'], gltf_loader) if ('baseColorTexture' in pbr_metallic_roughness_entry) else None
        self._metallic_roughness_texture = Texture(pbr_metallic_roughness_entry['metallicRoughnessTexture'], gltf_loader) if ('metallicRoughnessTexture' in pbr_metallic_roughness_entry) else None

    def get_base_color_texture(self):
        return self._base_color_texture

    def get_base_color_factor(self):
        return self._base_color_factor

    def get_metallic_roughness_texture(self):
        return self._metallic_roughness_texture

    def get_metallic_factor(self):
        return self._metallic_factor

    def get_roughness_factor(self):
        return self._roughness_factor
        
class PbrSpecularGlossiness(object):
    def __init__(self, pbr_specular_glossiness_entry, gltf_loader):
        self._diffuse_factor = pbr_specular_glossiness_entry['diffuseFactor'] if ('diffuseFactor' in pbr_specular_glossiness_entry) else [1.0,1.0,1.0,1.0]
        self._diffuse_texture = Texture(pbr_specular_glossiness_entry['diffuseTexture'], gltf_loader) if ('diffuseTexture' in pbr_specular_glossiness_entry) else None
        self._specular_factor = pbr_specular_glossiness_entry['specularFactor'] if ('specularFactor' in pbr_specular_glossiness_entry) else [1.0,1.0,1.0]
        self._glossiness_factor = pbr_specular_glossiness_entry['glossinessFactor'] if ('glossinessFactor' in pbr_specular_glossiness_entry) else 1.0
        self._specular_glossiness_texture = Texture(pbr_specular_glossiness_entry['specularGlossinessTexture'], gltf_loader) if ('specularGlossinessTexture' in pbr_specular_glossiness_entry) else None

    def get_diffuse_factor(self):
        return self._diffuse_factor

    def get_specular_glossiness_texture(self):
        return self._specular_glossiness_texture

    def get_specular_factor(self):
        return self._specular_factor

    def get_glossiness_factor(self):
        return self._glossiness_factor

    def get_diffuse_texture(self):
        return self._diffuse_texture

class KHRTextureTransform(object):
    def __init__(self, khr_texture_transform_entry):
        self._offset = khr_texture_transform_entry['offset'] if 'offset' in khr_texture_transform_entry else [0.0,0.0]
        self._scale = khr_texture_transform_entry['scale'] if 'scale' in khr_texture_transform_entry else [1.0,1.0]
        self._rotation = khr_texture_transform_entry['rotation'] if 'rotation' in khr_texture_transform_entry else 0.0

    @property
    def offset(self):
        return self._offset

    @property
    def scale(self):
        return self._scale

    @property
    def rotation(self):
        return self._rotation


class Material:
    def __init__(self, material_entry, material_index, gltf_loader):
        self._name = material_entry['name'] if ('name' in material_entry) else 'material_{}'.format(material_index)
        self._index = material_index
        self._double_sided = material_entry['doubleSided'] if ('doubleSided' in material_entry) else False
        
        self._pbr_metallic_roughness = PbrMetallicRoughness(material_entry['pbrMetallicRoughness'], gltf_loader) if ('pbrMetallicRoughness' in material_entry) else None

        self._alpha_mode = material_entry['alphaMode'] if ('alphaMode' in material_entry) else AlphaMode.OPAQUE

        self._normal_texture = NormalTexture(material_entry['normalTexture'], gltf_loader) if ('normalTexture' in material_entry) else None
        self._emissive_factor = material_entry['emissiveFactor'] if ('emissiveFactor' in material_entry) else [0,0,0]
        self._emissive_texture = Texture(material_entry['emissiveTexture'], gltf_loader) if ('emissiveTexture' in material_entry) else None
        self._occlusion_texture = OcclusionTexture(material_entry['occlusionTexture'], gltf_loader) if ('occlusionTexture' in material_entry) else None

        self._extensions = {}
        if 'extensions' in material_entry and 'KHR_materials_pbrSpecularGlossiness' in material_entry['extensions']:
            self._extensions['KHR_materials_pbrSpecularGlossiness'] = PbrSpecularGlossiness(material_entry['extensions']['KHR_materials_pbrSpecularGlossiness'], gltf_loader)

    def is_double_sided(self):
        return self._double_sided

    @property
    def alpha_mode(self):
        return self._alpha_mode

    def get_index(self):
        return self._index

    def get_pbr_metallic_roughness(self):
        return self._pbr_metallic_roughness

    def get_name(self):
        return self._name

    def get_extensions(self):
        return self._extensions

    def get_normal_texture(self):
        return self._normal_texture

    def get_occlusion_texture(self):
        return self._occlusion_texture

    def get_emissive_texture(self):
        return self._emissive_texture

    def get_emissive_factor(self):
        return self._emissive_factor