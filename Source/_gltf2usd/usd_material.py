from enum import Enum

from pxr import Gf, Sdf, UsdGeom, UsdShade

from gltf2 import Material, GLTFImage
from gltf2.Material import AlphaMode
from gltf2loader import TextureWrap

class USDMaterial():
    def __init__(self, stage, material_scope, index, gltf2loader):
        name = 'pbrmaterial_{}'.format(index)
        self._gltf2loader = gltf2loader
        self._stage = stage
        self._material_scope = material_scope
        self._material_path = Sdf.Path('{0}/{1}'.format('/Materials', name))
        self._usd_material = UsdShade.Material.Define(stage, self._material_path)
        
        self._usd_material_surface_output = self._usd_material.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        self._usd_material_displacement_output = self._usd_material.CreateOutput("displacement", Sdf.ValueTypeNames.Token)

    def convert_material_to_usd_preview_surface(self, gltf_material, output_directory):
        usd_preview_surface = USDPreviewSurface(self._stage, gltf_material, self, output_directory)
        usd_preview_surface._name = gltf_material.get_name()

    def get_usd_material(self):
        return self._usd_material




class USDPreviewSurface():
    """Models a physically based surface for USD
    """
    def __init__(self, stage, gltf_material, usd_material, output_directory):
        self._stage = stage
        self._usd_material = usd_material
        self._output_directory = output_directory
        material_path = usd_material._usd_material.GetPath()
        material = UsdShade.Shader.Define(self._stage, material_path.AppendChild('pbrMat'))
        material.CreateIdAttr('UsdPreviewSurface')
        self._initialize_material(material, self)
        self._initialize_from_gltf_material(gltf_material)

        self._stage = stage


    def _initialize_material(self, material, usd_preview_surface_material):
        shader = material
        self._use_specular_workflow = material.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Int)
        self._use_specular_workflow.Set(False)

        self._surface_output = shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
        self._usd_material._usd_material_surface_output.ConnectToSource(self._surface_output)
       

        self._displacement_output = shader.CreateOutput('displacement', Sdf.ValueTypeNames.Token)
        self._usd_material._usd_material_displacement_output.ConnectToSource(self._displacement_output)
        

        self._diffuse_color = (1.0, 1.0, 1.0)
        self._emissive_color = (0.0, 0.0, 0.0)
        self._specular_color = material.CreateInput('specularColor', Sdf.ValueTypeNames.Color3f)
        self._specular_color.Set((1.0,1.0,1.0))
        
        self._metallic = material.CreateInput('metallic', Sdf.ValueTypeNames.Float)
        self._metallic.Set(1.0)

        self._roughness = material.CreateInput('roughness', Sdf.ValueTypeNames.Float)
        self._roughness.Set(1.0)

        self._clearcoat = material.CreateInput('clearcoat', Sdf.ValueTypeNames.Float)
        self._clearcoat.Set(0.0)

        self._clearcoat_roughness = material.CreateInput('clearcoatRoughness', Sdf.ValueTypeNames.Float)
        self._clearcoat_roughness.Set(0.01)

        self._opacity = material.CreateInput('opacity', Sdf.ValueTypeNames.Float)
        self._opacity.Set(1.0)

        self._ior = material.CreateInput('ior', Sdf.ValueTypeNames.Float)
        self._ior.Set(1.5)

        self._normal = material.CreateInput('normal', Sdf.ValueTypeNames.Normal3f)

        self._displacement = material.CreateInput('displacement', Sdf.ValueTypeNames.Float)
        self._displacement.Set(0.0)

        self._occlusion = material.CreateInput('occlusion', Sdf.ValueTypeNames.Float)
        self._occlusion.Set(1.0)

        self._emissive_color = material.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f)
        self._emissive_color.Set((0, 0, 0))

        self._diffuse_color = material.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f)
        self._diffuse_color.Set((1, 1, 1))

        self._st0 = USDPrimvarReaderFloat2(self._stage, self._usd_material._material_path, 'st0')
        self._st1 = USDPrimvarReaderFloat2(self._stage, self._usd_material._material_path, 'st1')


    def _initialize_from_gltf_material(self, gltf_material):
        self._set_normal_texture(gltf_material)
        self._set_emissive_texture(gltf_material)
        self._set_occlusion_texture(gltf_material)
        self._set_khr_material_pbr_specular_glossiness(gltf_material)
        
        

    def _set_normal_texture(self, gltf_material):
        normal_texture = gltf_material.get_normal_texture()
        if (not normal_texture):
            self._normal.Set((0,0,1))
        else:
            destination = normal_texture.write_to_directory(self._output_directory, GLTFImage.ImageColorChannels.RGB)
            normal_scale = normal_texture.scale
            scale_factor = (normal_scale, normal_scale, normal_scale, 1.0)
            usd_uv_texture = USDUVTexture("normalTexture", self._stage, self._usd_material._usd_material, normal_texture, [self._st0, self._st1])
            usd_uv_texture._file_asset.Set(destination)
            usd_uv_texture._scale.Set(scale_factor)
            usd_uv_texture._fallback.Set(scale_factor)
            texture_shader = usd_uv_texture.get_shader()
            texture_shader.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
            self._normal.ConnectToSource(texture_shader, 'rgb')

    def _set_emissive_texture(self, gltf_material):
        emissive_texture = gltf_material.get_emissive_texture()
        emissive_factor = gltf_material.get_emissive_factor()
        if (not emissive_texture):
            self._emissive_color.Set((0,0,0))
        else:
            destination = emissive_texture.write_to_directory(self._output_directory, GLTFImage.ImageColorChannels.RGB)
            scale_factor = (emissive_factor[0], emissive_factor[1], emissive_factor[2], 1.0)
            usd_uv_texture = USDUVTexture("emissiveTexture", self._stage, self._usd_material._usd_material, emissive_texture, [self._st0, self._st1])
            usd_uv_texture._file_asset.Set(destination)
            usd_uv_texture._scale.Set(scale_factor)
            usd_uv_texture._fallback.Set(scale_factor)
            texture_shader = usd_uv_texture.get_shader()
            texture_shader.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
            self._emissive_color.ConnectToSource(texture_shader, 'rgb')

    def _set_occlusion_texture(self, gltf_material):
        occlusion_texture = gltf_material.get_occlusion_texture()      
        if (not occlusion_texture):
            self._occlusion.Set(1.0)
        else:
            destination = occlusion_texture.write_to_directory(self._output_directory, GLTFImage.ImageColorChannels.R)
            occlusion_strength = occlusion_texture.strength
            strength_factor = (occlusion_strength, occlusion_strength, occlusion_strength, 1.0)
            usd_uv_texture = USDUVTexture("occlusionTexture", self._stage, self._usd_material._usd_material, occlusion_texture, [self._st0, self._st1])
            usd_uv_texture._file_asset.Set(destination)
            usd_uv_texture._scale.Set(strength_factor)
            usd_uv_texture._fallback.Set(strength_factor)
            texture_shader = usd_uv_texture.get_shader()
            texture_shader.CreateOutput('r', Sdf.ValueTypeNames.Float)
            self._occlusion.ConnectToSource(texture_shader, 'r')

    def _set_pbr_metallic_roughness(self, gltf_material):
        pbr_metallic_roughness = gltf_material.get_pbr_metallic_roughness()
        if (pbr_metallic_roughness):
            self._set_pbr_base_color(pbr_metallic_roughness, gltf_material.alpha_mode)
            self._set_pbr_metallic(pbr_metallic_roughness)
            self._set_pbr_roughness(pbr_metallic_roughness)

    def _set_khr_material_pbr_specular_glossiness(self, gltf_material):
        extensions = gltf_material.get_extensions()
        if not 'KHR_materials_pbrSpecularGlossiness' in extensions:
            self._set_pbr_metallic_roughness(gltf_material)
        else:
            self._use_specular_workflow.Set(True)
            pbr_specular_glossiness = extensions['KHR_materials_pbrSpecularGlossiness']
            self._set_pbr_specular_glossiness_diffuse(pbr_specular_glossiness)
            self._set_pbr_specular_glossiness_glossiness(pbr_specular_glossiness)
            self._set_pbr_specular_glossiness_specular(pbr_specular_glossiness)

    def _set_pbr_specular_glossiness_diffuse(self, pbr_specular_glossiness):
        diffuse_texture = pbr_specular_glossiness.get_diffuse_texture()
        diffuse_factor = pbr_specular_glossiness.get_diffuse_factor()
        if not diffuse_texture:
            self._diffuse_color.Set(Gf.Vec3f(diffuse_factor[0], diffuse_factor[1], diffuse_factor[2]))
        else:
            destination = diffuse_texture.write_to_directory(self._output_directory, GLTFImage.ImageColorChannels.RGB)
            scale_factor = tuple(diffuse_factor)
            usd_uv_texture = USDUVTexture("diffuseTexture", self._stage, self._usd_material._usd_material, diffuse_texture, [self._st0, self._st1])
            usd_uv_texture._file_asset.Set(destination)
            usd_uv_texture._scale.Set(scale_factor)
            usd_uv_texture._fallback.Set(scale_factor)
            texture_shader = usd_uv_texture.get_shader()
            texture_shader.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
            self._diffuse_color.ConnectToSource(texture_shader, 'rgb')

    def _set_pbr_specular_glossiness_specular(self, pbr_specular_glossiness):
        specular_glossiness_texture = pbr_specular_glossiness.get_specular_glossiness_texture()
        
        specular_factor = tuple(pbr_specular_glossiness.get_specular_factor())
        if not specular_glossiness_texture:
            self._specular_color.Set(specular_factor)
        else:
            destination = specular_glossiness_texture.write_to_directory(self._output_directory, GLTFImage.ImageColorChannels.RGB, "specular")
            scale_factor = (specular_factor[0], specular_factor[1], specular_factor[2], 1) 
            usd_uv_texture = USDUVTexture("specularTexture", self._stage, self._usd_material._usd_material, specular_glossiness_texture, [self._st0, self._st1])
            usd_uv_texture._file_asset.Set(destination)
            usd_uv_texture._scale.Set(scale_factor)
            usd_uv_texture._fallback.Set(scale_factor)
            texture_shader = usd_uv_texture.get_shader()
            texture_shader.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
            self._specular_color.ConnectToSource(texture_shader, 'rgb')

    def _set_pbr_specular_glossiness_glossiness(self, pbr_specular_glossiness):
        specular_glossiness_texture = pbr_specular_glossiness.get_specular_glossiness_texture()
        roughness_factor = 1 - pbr_specular_glossiness.get_glossiness_factor()
        if not specular_glossiness_texture:
            self._roughness.Set(roughness_factor)
        else:
            destination = specular_glossiness_texture.write_to_directory(self._output_directory, GLTFImage.ImageColorChannels.A, "glossiness")
            scale_factor = (-1, -1, -1, -1)
            usd_uv_texture = USDUVTexture("glossinessTexture", self._stage, self._usd_material._usd_material, specular_glossiness_texture, [self._st0, self._st1])
            usd_uv_texture._file_asset.Set(destination)
            usd_uv_texture._bias.Set((1.0, 1.0, 1.0, 1.0))
            usd_uv_texture._scale.Set(scale_factor)
            usd_uv_texture._fallback.Set(scale_factor)
            texture_shader = usd_uv_texture.get_shader()
            texture_shader.CreateOutput('r', Sdf.ValueTypeNames.Float)
            self._roughness.ConnectToSource(texture_shader, 'r')

            

    def _set_pbr_base_color(self, pbr_metallic_roughness, alpha_mode):
        base_color_texture = pbr_metallic_roughness.get_base_color_texture()
        base_color_scale = pbr_metallic_roughness.get_base_color_factor()
        if AlphaMode(alpha_mode) != AlphaMode.OPAQUE:
            if AlphaMode(alpha_mode) == AlphaMode.MASK:
                print('Alpha Mask not supported in USDPreviewSurface!  Using Alpha Blend...')

            self._opacity.Set(base_color_scale[3])

        if not base_color_texture:
            self._diffuse_color.Set(tuple(base_color_scale[0:3]))
        else:
            if AlphaMode(alpha_mode) == AlphaMode.OPAQUE:
                destination = base_color_texture.write_to_directory(self._output_directory, GLTFImage.ImageColorChannels.RGB)
                scale_factor = (base_color_scale[0], base_color_scale[1], base_color_scale[2], 1.0)
            else:
                destination = base_color_texture.write_to_directory(self._output_directory, GLTFImage.ImageColorChannels.RGBA)
                scale_factor = tuple(base_color_scale[0:4])
            usd_uv_texture = USDUVTexture("baseColorTexture", self._stage, self._usd_material._usd_material, base_color_texture, [self._st0, self._st1])
            usd_uv_texture._file_asset.Set(destination)
            usd_uv_texture._scale.Set(scale_factor)
            usd_uv_texture._fallback.Set(scale_factor)
            texture_shader = usd_uv_texture.get_shader()
            texture_shader.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
            self._diffuse_color.ConnectToSource(texture_shader, 'rgb')

    def _set_pbr_metallic(self, pbr_metallic_roughness):
        metallic_roughness_texture = pbr_metallic_roughness.get_metallic_roughness_texture()
        metallic_factor = pbr_metallic_roughness.get_metallic_factor()
        if not metallic_roughness_texture:
            self._metallic.Set(metallic_factor)
        else:
            destination = metallic_roughness_texture.write_to_directory(self._output_directory, GLTFImage.ImageColorChannels.B, "metallic")
            scale_factor = tuple([metallic_factor]*4)
            usd_uv_texture = USDUVTexture("metallicTexture", self._stage, self._usd_material._usd_material, metallic_roughness_texture, [self._st0, self._st1])
            usd_uv_texture._file_asset.Set(destination)
            usd_uv_texture._scale.Set(scale_factor)
            usd_uv_texture._fallback.Set(scale_factor)
            texture_shader = usd_uv_texture.get_shader()
            texture_shader.CreateOutput('r', Sdf.ValueTypeNames.Float)
            self._metallic.ConnectToSource(texture_shader, 'r')

    def _set_pbr_roughness(self, pbr_metallic_roughness):
        metallic_roughness_texture = pbr_metallic_roughness.get_metallic_roughness_texture()
        roughness_factor = pbr_metallic_roughness.get_roughness_factor()
        if not metallic_roughness_texture:
            self._roughness.Set(roughness_factor)
        else:
            destination = metallic_roughness_texture.write_to_directory(self._output_directory, GLTFImage.ImageColorChannels.G, "roughness")
            scale_factor = tuple([roughness_factor]*4)
            usd_uv_texture = USDUVTexture("roughnessTexture", self._stage, self._usd_material._usd_material, metallic_roughness_texture, [self._st0, self._st1])
            usd_uv_texture._file_asset.Set(destination)
            usd_uv_texture._scale.Set(scale_factor)
            usd_uv_texture._fallback.Set(scale_factor)
            texture_shader = usd_uv_texture.get_shader()
            texture_shader.CreateOutput('r', Sdf.ValueTypeNames.Float)
            self._roughness.ConnectToSource(texture_shader, 'r')





        


    def export_to_stage(self, usd_material):
        """Converts a glTF material to a usd preview surface

        Arguments:
            gltf_material {Material} -- glTF Material
        """
        material = UsdShade.Shader.Define(name, usd_material._stage, usd_material._material_path.AppendChild(self._name))
        material.CreateIdAttr('UsdPreviewSurface')
        material.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Int).Set(self._use_specular_workflow)
        surface_output = material.CreateOutput('surface', Sdf.ValueTypeNames.Token)
        usd_material._usd_material_surface_output.ConnectToSource(surface_output)
        displacement_output = material.CreateOutput('displacement', Sdf.ValueTypeNames.Token)
        usd_material._usd_material_displacement_output.ConnectToSource(displacement_output)


class USDPrimvarReaderFloat2():
    def __init__(self, stage, material_path, var_name):
        primvar = UsdShade.Shader.Define(stage, material_path.AppendChild('primvar_{}'.format(var_name)))
        primvar.CreateIdAttr('UsdPrimvarReader_float2')
        primvar.CreateInput('fallback', Sdf.ValueTypeNames.Float2).Set((0,0))
        primvar.CreateInput('varname', Sdf.ValueTypeNames.Token).Set(var_name)
        self._output = primvar.CreateOutput('result', Sdf.ValueTypeNames.Float2)

    def get_output(self):
        return self._output



class USDUVTextureWrapMode(Enum):
    BLACK = 'black'
    CLAMP = 'clamp'
    REPEAT = 'repeat'
    MIRROR = 'mirror'


class USDUVTexture(object):
    TEXTURE_SAMPLER_WRAP = {
            TextureWrap.CLAMP_TO_EDGE.name : 'clamp',
            TextureWrap.MIRRORED_REPEAT.name : 'mirror',
            TextureWrap.REPEAT.name: 'repeat',
        }
    def __init__(self, name, stage, usd_material, gltf_texture, usd_primvar_st_arr):
        
        material_path = usd_material.GetPath()

        self._texture_shader = UsdShade.Shader.Define(stage, material_path.AppendChild(name))
        self._texture_shader.CreateIdAttr("UsdUVTexture")

        self._wrap_s = self._texture_shader.CreateInput('wrapS', Sdf.ValueTypeNames.Token)
        self._wrap_s.Set(USDUVTexture.TEXTURE_SAMPLER_WRAP[gltf_texture.get_wrap_s().name])

        self._wrap_t =self._texture_shader.CreateInput('wrapT', Sdf.ValueTypeNames.Token)
        self._wrap_t.Set(USDUVTexture.TEXTURE_SAMPLER_WRAP[gltf_texture.get_wrap_t().name])

        self._bias = self._texture_shader.CreateInput('bias', Sdf.ValueTypeNames.Float4)
        self._bias.Set((0,0,0,0))

        self._scale = self._texture_shader.CreateInput('scale', Sdf.ValueTypeNames.Float4)
        self._scale.Set((1,1,1,1))

        self._file_asset = self._texture_shader.CreateInput('file', Sdf.ValueTypeNames.Asset)
        self._file_asset.Set(gltf_texture.get_image_path())

        self._fallback = self._texture_shader.CreateInput('fallback', Sdf.ValueTypeNames.Float4)
        self._fallback.Set((0,0,0,1))

        self._st = self._texture_shader.CreateInput('st', Sdf.ValueTypeNames.Float2)

        self._st.ConnectToSource(usd_primvar_st_arr[gltf_texture.get_texcoord_index()].get_output())


    def get_shader(self):
        return self._texture_shader




