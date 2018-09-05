import argparse
import base64
import collections
import json
import logging
import ntpath
import numpy
import os
import re
import shutil

from io import BytesIO

from gltf2loader import GLTF2Loader, PrimitiveMode, TextureWrap, MinFilter, MagFilter

from PIL import Image

from pxr import Usd, UsdGeom, Sdf, UsdShade, Gf, UsdSkel, Vt

AnimationsMap = collections.namedtuple('AnimationMap', ('path', 'sampler'))
Node = collections.namedtuple('Node', ('index', 'parent', 'children', 'name', 'hierarchy_name'))
KeyFrame = collections.namedtuple('KeyFrame', ('input', 'output'))

class JointData:
    def __init__(self, skeleton_joint, joint_name, joint_index):
        self.skeleton_joint = skeleton_joint
        self.joint_name = joint_name
        self.joint_index = joint_index

class GLTF2USD:
    """
    Class for converting glTF 2.0 models to Pixar's USD format.  Currently openly supports .gltf files
    with non-embedded data and exports to .usda .
    """

    TEXTURE_SAMPLER_WRAP = {
        TextureWrap.CLAMP_TO_EDGE : 'clamp',
        TextureWrap.MIRRORED_REPEAT : 'mirror',
        TextureWrap.REPEAT: 'repeat',
    }

    def __init__(self, gltf_file, usd_file, fps, scale, verbose=False):
        """Initializes the glTF to USD converter

        Arguments:
            gltf_file {str} -- path to the glTF file
            usd_file {str} -- path to store the generated usda file
            verbose {boolean} -- specifies if the output should be verbose from this tool
        """
        self.logger = logging.getLogger('gltf2usd')
        self.logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)

        if not usd_file.endswith('.usda'):
            self.logger.error('This tool can only export to .usda file format')
        else:
            self.fps = fps

            self.gltf_loader = GLTF2Loader(gltf_file)
            self.verbose = verbose
            self.scale = scale

            self.output_dir = os.path.dirname(usd_file)

            self.stage = Usd.Stage.CreateNew(usd_file)
            self.gltf_usd_nodemap = {}
            self.gltf_usdskel_nodemap = {}
            self._usd_mesh_skin_map = {}

            self.convert()


    def _get_child_nodes(self):
        """
        Returns all the child nodes in the glTF scene hierarchy as a set
        """
        child_nodes = set()
        for node in self.gltf_loader.json_data['nodes']:
            if 'children' in node:
                child_nodes.update(node['children'])

        return child_nodes


    def convert_nodes_to_xform(self):
        """
        Converts the glTF nodes to USD Xforms.  The models get a parent Xform that scales the geometry by 100
        to convert from meters (glTF) to centimeters (USD).
        """
        parent_transform = UsdGeom.Xform.Define(self.stage, '/root')
        parent_transform.AddScaleOp().Set((self.scale, self.scale, self.scale))

        self.node_hierarchy = self._build_node_hierarchy()

        child_nodes = self._get_child_nodes()
        if 'scenes' in self.gltf_loader.json_data:
            main_scene = self.gltf_loader.json_data['scene'] if 'scene' in self.gltf_loader.json_data else 0
            child_nodes = self._get_child_nodes()
            for i, node_index in enumerate(self.gltf_loader.json_data['scenes'][main_scene]['nodes']):
                node = self.gltf_loader.json_data['nodes'][node_index]

                if node_index not in child_nodes:
                    name = node['name'] if 'name' in node else 'node{}'.format(i)
                    xform_name = '{0}/{1}'.format(parent_transform.GetPath(), name)
                    self._convert_node_to_xform(node, node_index, xform_name)

            self._convert_animations_to_usd()
            self.stage.GetRootLayer().Save()

        self.logger.info('Conversion complete!')

    def _init_animations_map(self):
        """Creates a mapping of glTF node indices to sampler animations
        """

        if 'animations' in self.gltf_loader.json_data:
            self.animations_map = {}

            for animation in self.gltf_loader.json_data['animations']:
                if 'channels' in animation:
                    for channel in animation['channels']:
                        target = channel['target']
                        animation_map = AnimationsMap(path=target['path'], sampler=animation['samplers'][channel['sampler']])

                        if target['node'] in self.animations_map:
                            self.animations_map[target['node']].append(animation_map)
                        else:
                            self.animations_map[target['node']] = [animation_map]


    def _convert_node_to_xform(self, node, node_index, xform_name):
        """Converts a glTF node to a USD transform node.

        Arguments:
            node {dict} -- glTF node
            node_index {int} -- glTF node index
            xform_name {str} -- USD xform name
        """
        xformPrim = UsdGeom.Xform.Define(self.stage, xform_name)
        self.gltf_usd_nodemap[node_index] = xformPrim

        xform_matrix = self._compute_rest_matrix(node)
        xformPrim.AddTransformOp().Set(xform_matrix)

        if 'mesh' in node:
            skin_index = node['skin'] if 'skin' in node else None

            # each mesh gets it's own SkelRoot
            skeleton_path = '{0}/{1}'.format(xform_name, 'skeleton_root')
            skel_root = UsdSkel.Root.Define(self.stage, skeleton_path)
            usd_parent_node = skel_root

            mesh = self.gltf_loader.json_data['meshes'][node['mesh']]
            self._convert_mesh_to_xform(mesh, usd_parent_node, node_index, skin_index)

        if 'children' in node:
            for child_index in node['children']:
                child_node = self.gltf_loader.json_data['nodes'][child_index]
                child_name = child_node['name'] if 'name' in node else 'node{}'.format(child_index)
                child_xform_name = '{0}/{1}'.format(xform_name, child_name)
                self._convert_node_to_xform(child_node, child_index, child_xform_name)


    def _convert_mesh_to_xform(self, mesh, usd_parent_node, node_index, skin_index=None):
        """
        Converts a glTF mesh to a USD Xform.
        Each primitive becomes a submesh of the Xform.

        Arguments:
            mesh {dict} -- glTF mesh
            parent_path {str} -- xform path
            node_index {int} -- glTF node index
        """
        #for each mesh primitive, create a USD mesh
        if 'primitives' in mesh:
            for i, mesh_primitive in enumerate(mesh['primitives']):
                mesh_primitive_name = 'mesh_primitive{}'.format(i)
                double_sided = False
                if 'material' in mesh_primitive and 'doubleSided' in self.gltf_loader.json_data['materials'][mesh_primitive['material']]:
                    double_sided = self.gltf_loader.json_data['materials'][mesh_primitive['material']]['doubleSided']
                self._convert_primitive_to_mesh(
                    name=mesh_primitive_name,
                    primitive=mesh_primitive,
                    usd_parent_node=usd_parent_node,
                    node_index=node_index,
                    double_sided = double_sided,
                    skin_index=skin_index)


    def _convert_primitive_to_mesh(self, name, primitive, usd_parent_node, node_index, double_sided, skin_index):
        """
        Converts a glTF mesh primitive to a USD mesh

        Arguments:
            name {str} -- name of the primitive
            primitive {dict} -- glTF primitive
            usd_parent_node {str} -- USD parent xform
            node_index {int} -- glTF node index
            double_sided {bool} -- specifies if the primitive is double sided
        """
        parent_path = usd_parent_node.GetPath()
        usd_node = self.stage.GetPrimAtPath(parent_path)
        gltf_node = self.gltf_loader.json_data['nodes'][node_index]
        gltf_mesh = self.gltf_loader.json_data['meshes'][gltf_node['mesh']]
        mesh = UsdGeom.Mesh.Define(self.stage, '{0}/{1}'.format(parent_path, name))
        mesh.CreateSubdivisionSchemeAttr().Set('none')

        if double_sided:
            mesh.CreateDoubleSidedAttr().Set(double_sided)

        if 'material' in primitive:
            usd_material = self.usd_materials[primitive['material']]
            UsdShade.MaterialBindingAPI(mesh).Bind(usd_material)
        if 'attributes' in primitive:
            for attribute in primitive['attributes']:
                if attribute == 'POSITION':
                    accessor_index = primitive['attributes'][attribute]
                    accessor = self.gltf_loader.json_data['accessors'][accessor_index]
                    override_prim = self.stage.OverridePrim(mesh.GetPath())

                    override_prim.CreateAttribute('extent', Sdf.ValueTypeNames.Float3Array).Set([accessor['min'], accessor['max']])
                    data = self.gltf_loader.get_data(accessor=accessor)
                    mesh.CreatePointsAttr(data)
                if attribute == 'NORMAL':
                    accessor_index = primitive['attributes'][attribute]
                    accessor = self.gltf_loader.json_data['accessors'][accessor_index]
                    data = self.gltf_loader.get_data(accessor=accessor)
                    mesh.CreateNormalsAttr(data)

                if attribute == 'COLOR_0':
                    accessor_index = primitive['attributes'][attribute]
                    accessor = self.gltf_loader.json_data['accessors'][accessor_index]
                    data = self.gltf_loader.get_data(accessor=accessor)
                    prim_var = UsdGeom.PrimvarsAPI(mesh)
                    colors = prim_var.CreatePrimvar('displayColor', Sdf.ValueTypeNames.Color3f, 'vertex').Set(data)

                if attribute == 'TEXCOORD_0':
                    accessor_index = primitive['attributes'][attribute]
                    accessor = self.gltf_loader.json_data['accessors'][accessor_index]
                    data = self.gltf_loader.get_data(accessor=accessor)
                    invert_uvs = []
                    for uv in data:
                        new_uv = (uv[0], 1 - uv[1])
                        invert_uvs.append(new_uv)
                    prim_var = UsdGeom.PrimvarsAPI(mesh)
                    uv = prim_var.CreatePrimvar('primvars:st0', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex')
                    uv.Set(invert_uvs)
                if attribute == 'JOINTS_0':
                    gltf_node = self.gltf_loader.json_data['nodes'][node_index]
                    self._convert_skin_to_usd(gltf_node, node_index, usd_parent_node, mesh)
                    if skin_index != None:
                        self._usd_mesh_skin_map[skin_index] = mesh
        if 'targets' in primitive:
            skinBinding = UsdSkel.BindingAPI.Apply(mesh.GetPrim())

            skeleton = UsdSkel.Skeleton.Define(self.stage, '{0}/skel'.format(parent_path))

            # Create an animation for this mesh to hold the blendshapes
            skelAnim = UsdSkel.Animation.Define(self.stage, '{0}/skel/anim'.format(parent_path))

            # link the skeleton animation to skelAnim
            skinBinding.CreateAnimationSourceRel().AddTarget(skelAnim.GetPath())

            # link the skeleton to skeleton
            skinBinding.CreateSkeletonRel().AddTarget(skeleton.GetPath())

            # Fetch the names and accessors of the blendshapes
            targets = primitive['targets']
            positions = list(map(lambda target: target['POSITION'], targets))
            accessors = list(map(lambda idx: self.gltf_loader.json_data['accessors'][idx], positions))

            # Set blendshape names on the animation
            names = []
            for i, weight in enumerate(gltf_mesh['weights']):
                accessor = accessors[i]
                blend_shape_name = accessor['name'] if 'name' in accessor else 'shape_{}'.format(i)
                names.append(blend_shape_name)

            skelAnim.CreateBlendShapesAttr().Set(names)
            skinBinding.CreateBlendShapesAttr(names)

            # Set the starting weights of each blendshape to the weights defined in the glTF primitive
            gltf_node = self.gltf_loader.json_data['nodes'][node_index]
            gltf_mesh = self.gltf_loader.json_data['meshes'][gltf_node['mesh']]
            weights = gltf_mesh['weights']
            skelAnim.CreateBlendShapeWeightsAttr().Set(weights)
            blend_shape_targets = skinBinding.CreateBlendShapeTargetsRel()

            # Define offsets for each blendshape, and add them as skel:blendShapes and skel:blendShapeTargets
            for i, name in enumerate(names):
                accessor = accessors[i]
                offsets = self.gltf_loader.get_data(buffer=self.buffer, accessor=accessor)
                blend_shape_name = '{0}/{1}'.format(mesh.GetPath(), name)

                # define blendshapes in the mesh
                blend_shape = UsdSkel.BlendShape.Define(self.stage, blend_shape_name)
                blend_shape.CreateOffsetsAttr(offsets)
                blend_shape_targets.AddTarget(name)

        if 'indices' in primitive:
            #TODO: Need to support glTF primitive modes.  Currently only Triangle mode is supported
            indices = self.gltf_loader.get_data(accessor=self.gltf_loader.json_data['accessors'][primitive['indices']])
            num_faces = len(indices)/3
            face_count = [3] * num_faces
            mesh.CreateFaceVertexCountsAttr(face_count)
            mesh.CreateFaceVertexIndicesAttr(indices)
        else:
            position_accessor = self.gltf_loader.json_data['accessors'][primitive['attributes']['POSITION']]
            count = position_accessor['count']
            num_faces = count/3
            indices = range(0, count)
            face_count = [3] * num_faces
            mesh.CreateFaceVertexCountsAttr(face_count)
            mesh.CreateFaceVertexIndicesAttr(indices)

        if 'material' in primitive:
            material = self.gltf_loader.json_data['materials'][primitive['material']]

    def _get_texture__wrap_modes(self, texture):
        """Get the USD texture wrap modes from a glTF texture

        Arguments:
            texture {dict} -- glTF texture

        Returns:
            dict -- dictionary mapping wrapS and wrapT to
            a USD texture sampler mode
        """

        texture_data = {'wrapS': 'repeat', 'wrapT': 'repeat'}
        if 'sampler' in texture:
            sampler = self.gltf_loader.json_data['samplers'][texture['sampler']]

            if 'wrapS' in sampler:
                texture_data['wrapS'] = GLTF2USD.TEXTURE_SAMPLER_WRAP[TextureWrap(sampler['wrapS'])]

            if 'wrapT' in sampler:
                texture_data['wrapT'] = GLTF2USD.TEXTURE_SAMPLER_WRAP[TextureWrap(sampler['wrapT'])]

        return texture_data

    def _convert_images_to_usd(self):
        """
        Converts the glTF images to USD images
        """

        if 'images' in self.gltf_loader.json_data:
            self.images = []
            for i, image in enumerate(self.gltf_loader.json_data['images']):
                image_name = ''

                # save data-uri textures
                if image['uri'].startswith('data:image'):
                    uri_data = image['uri'].split(',')[1]
                    img = Image.open(BytesIO(base64.b64decode(uri_data)))

                    # NOTE: image might not have a name
                    image_name = image['name'] if 'name' in image else 'image{}.{}'.format(i, img.format)
                    image_path = os.path.join(self.gltf_loader.root_dir, image_name)
                    img.save(image_path)

                # otherwise just copy the texture over
                else:
                    image_path = os.path.join(self.gltf_loader.root_dir, image['uri'])
                    image_name = os.path.join(self.output_dir, ntpath.basename(image_path))

                    if self.gltf_loader.root_dir is not self.output_dir:
                        shutil.copyfile(image_path, image_name)

                self.images.append(ntpath.basename(image_name))

    def _convert_materials_to_preview_surface(self):
        """
        Converts the glTF materials to preview surfaces
        """

        if 'materials' in self.gltf_loader.json_data:
            self.usd_materials = []
            material_path_root = '/Materials'
            scope = UsdGeom.Scope.Define(self.stage, material_path_root)

            for i, material in enumerate(self.gltf_loader.json_data['materials']):
                name = 'pbrmaterial_{}'.format(i)
                material_path = Sdf.Path('{0}/{1}'.format(material_path_root, name))
                usd_material = UsdShade.Material.Define(self.stage, material_path)
                self.usd_materials.append(usd_material)

                usd_material_surface_output = usd_material.CreateOutput("surface", Sdf.ValueTypeNames.Token)
                usd_material_displacement_output = usd_material.CreateOutput("displacement", Sdf.ValueTypeNames.Token)
                pbr_mat = UsdShade.Shader.Define(self.stage, material_path.AppendChild('pbrMat1'))
                pbr_mat.CreateIdAttr("UsdPreviewSurface")
                specular_workflow = pbr_mat.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Bool)
                specular_workflow.Set(False)
                pbr_mat_surface_output = pbr_mat.CreateOutput("surface", Sdf.ValueTypeNames.Token)
                pbr_mat_displacement_output = pbr_mat.CreateOutput("displacement", Sdf.ValueTypeNames.Token)
                usd_material_surface_output.ConnectToSource(pbr_mat_surface_output)
                usd_material_displacement_output.ConnectToSource(pbr_mat_displacement_output)

                #define uv primvar0
                primvar_st0 = UsdShade.Shader.Define(self.stage, material_path.AppendChild('primvar_st0'))
                primvar_st0.CreateIdAttr('UsdPrimvarReader_float2')
                fallback_st0 = primvar_st0.CreateInput('fallback', Sdf.ValueTypeNames.Float2)
                fallback_st0.Set((0,0))
                primvar_st0_varname = primvar_st0.CreateInput('varname', Sdf.ValueTypeNames.Token)
                primvar_st0_varname.Set('st0')
                primvar_st0_output = primvar_st0.CreateOutput('result', Sdf.ValueTypeNames.Float2)

                #define uv primvar1
                primvar_st1 = UsdShade.Shader.Define(self.stage, material_path.AppendChild('primvar_st1'))
                primvar_st1.CreateIdAttr('UsdPrimvarReader_float2')
                fallback_st1 = primvar_st1.CreateInput('fallback', Sdf.ValueTypeNames.Float2)
                fallback_st1.Set((0,0))
                primvar_st1_varname = primvar_st1.CreateInput('varname', Sdf.ValueTypeNames.Token)
                primvar_st1_varname.Set('st1')
                primvar_st1_output = primvar_st1.CreateOutput('result', Sdf.ValueTypeNames.Float2)

                pbr_metallic_roughness = None
                pbr_specular_glossiness = None

                if 'pbrMetallicRoughness' in material:
                    pbr_metallic_roughness = material['pbrMetallicRoughness']
                    if 'baseColorFactor' in pbr_metallic_roughness:
                        diffuse_color = pbr_mat.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
                        base_color_factor = pbr_metallic_roughness['baseColorFactor']
                        diffuse_color.Set((base_color_factor[0],base_color_factor[1],base_color_factor[2]))
                        opacity = pbr_mat.CreateInput("opacity", Sdf.ValueTypeNames.Float)
                        opacity.Set(base_color_factor[3])
                    if 'metallicFactor' in pbr_metallic_roughness:
                        metallic_factor = pbr_metallic_roughness['metallicFactor']
                        metallic = pbr_mat.CreateInput('metallic', Sdf.ValueTypeNames.Float)
                        metallic.Set(pbr_metallic_roughness['metallicFactor'])
                elif 'extensions' in material and 'KHR_materials_pbrSpecularGlossiness' in material['extensions']:
                    specular_workflow.Set(True)
                    pbr_specular_glossiness = material['extensions']['KHR_materials_pbrSpecularGlossiness']

                    specular_factor = pbr_specular_glossiness['specularFactor'] if ('specularFactor' in pbr_specular_glossiness) else (1,1,1)
                    pbr_mat.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((specular_factor[0], specular_factor[1], specular_factor[2]))
                    diffuse_factor = pbr_specular_glossiness['diffuseFactor'] if ('diffuseFactor' in pbr_specular_glossiness) else (1,1,1,1)
                    diffuse_color = pbr_mat.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
                    diffuse_color.Set((diffuse_factor[0], diffuse_factor[1], diffuse_factor[2]))
                    opacity = pbr_mat.CreateInput("opacity", Sdf.ValueTypeNames.Float)
                    opacity.Set(diffuse_factor[3])

                if 'occlusionTexture' in material:
                    occlusion_texture = material['occlusionTexture']
                    scale_factor = occlusion_texture['strength'] if 'strength' in occlusion_texture else 1
                    fallback_occlusion_value = scale_factor
                    scale_factor = (scale_factor, scale_factor, scale_factor, 1)
                    occlusion_components = {
                        'r':
                        {'sdf_type' : Sdf.ValueTypeNames.Float, 'name': 'occlusion'}
                    }

                    self._convert_texture_to_usd(
                        pbr_mat=pbr_mat,
                        gltf_texture=occlusion_texture,
                        gltf_texture_name= 'occlusionTexture',
                        color_components= occlusion_components,
                        scale_factor=scale_factor,
                        fallback_factor=fallback_occlusion_value,
                        material_path=material_path,
                        fallback_type=Sdf.ValueTypeNames.Float,
                        primvar_st0_output=primvar_st0_output,
                        primvar_st1_output=primvar_st1_output
                    )


                if 'normalTexture' in material:
                    normal_texture = material['normalTexture']
                    scale_factor = normal_texture['scale'] if 'scale' in normal_texture else 1
                    fallback_normal_color = (0,0,scale_factor)
                    scale_factor = (scale_factor, scale_factor, scale_factor, 1)
                    normal_components = {
                        'rgb':
                        {'sdf_type' : Sdf.ValueTypeNames.Normal3f, 'name': 'normal'}
                    }

                    self._convert_texture_to_usd(
                        pbr_mat=pbr_mat,
                        gltf_texture=material['normalTexture'],
                        gltf_texture_name='normalTexture',
                        color_components=normal_components,
                        scale_factor=scale_factor,
                        fallback_factor=fallback_normal_color,
                        material_path=material_path,
                        fallback_type=Sdf.ValueTypeNames.Normal3f,
                        primvar_st0_output=primvar_st0_output,
                        primvar_st1_output=primvar_st1_output
                    )

                if 'emissiveTexture' in material:
                    emissive_factor = material['emissiveFactor'] if 'emissiveFactor' in material else [0,0,0]
                    fallback_emissive_color = tuple(emissive_factor[0:3])
                    scale_emissive_factor = (emissive_factor[0], emissive_factor[1], emissive_factor[2], 1)
                    emissive_components = {
                        'rgb':
                        {'sdf_type' : Sdf.ValueTypeNames.Color3f, 'name': 'emissiveColor'}
                    }

                    self._convert_texture_to_usd(
                        pbr_mat=pbr_mat,
                        gltf_texture=material['emissiveTexture'],
                        gltf_texture_name='emissiveTexture',
                        color_components=emissive_components,
                        scale_factor=scale_emissive_factor,
                        fallback_factor=fallback_emissive_color,
                        material_path=material_path,
                        fallback_type=Sdf.ValueTypeNames.Color3f,
                        primvar_st0_output=primvar_st0_output,
                        primvar_st1_output=primvar_st1_output
                    )

                if pbr_specular_glossiness and 'diffuseTexture' in pbr_specular_glossiness:
                    base_color_factor = pbr_specular_glossiness['diffuseFactor'] if 'diffuseFactor' in pbr_specular_glossiness else (1,1,1,1)
                    fallback_base_color = (base_color_factor[0], base_color_factor[1], base_color_factor[2])
                    scale_base_color_factor = base_color_factor
                    base_color_components = {
                        'rgb':
                        {'sdf_type' : Sdf.ValueTypeNames.Color3f, 'name': 'diffuseColor'}
                    }

                    self._convert_texture_to_usd(
                        pbr_mat=pbr_mat,
                        gltf_texture=pbr_specular_glossiness['diffuseTexture'],
                        gltf_texture_name='diffuseTexture',
                        color_components=base_color_components,
                        scale_factor=scale_base_color_factor,
                        fallback_factor=fallback_base_color,
                        material_path=material_path,
                        fallback_type=Sdf.ValueTypeNames.Color3f,
                        primvar_st0_output=primvar_st0_output,
                        primvar_st1_output=primvar_st1_output
                    )


                if pbr_metallic_roughness and 'baseColorTexture' in pbr_metallic_roughness:
                    base_color_factor = pbr_metallic_roughness['baseColorFactor'] if 'baseColorFactor' in pbr_metallic_roughness else (1,1,1,1)
                    fallback_base_color = (base_color_factor[0], base_color_factor[1], base_color_factor[2])
                    scale_base_color_factor = base_color_factor
                    base_color_components = {
                        'rgb':
                        {'sdf_type' : Sdf.ValueTypeNames.Color3f, 'name': 'diffuseColor'}
                    }

                    self._convert_texture_to_usd(
                        pbr_mat=pbr_mat,
                        gltf_texture=pbr_metallic_roughness['baseColorTexture'],
                        gltf_texture_name='baseColorTexture',
                        color_components=base_color_components,
                        scale_factor=scale_base_color_factor,
                        fallback_factor=fallback_base_color,
                        material_path=material_path,
                        fallback_type=Sdf.ValueTypeNames.Color3f,
                        primvar_st0_output=primvar_st0_output,
                        primvar_st1_output=primvar_st1_output
                    )

                if pbr_metallic_roughness and 'metallicRoughnessTexture' in pbr_metallic_roughness:
                    metallic_roughness_texture_file = os.path.join(self.gltf_loader.root_dir, self.gltf_loader.json_data['images'][pbr_metallic_roughness['metallicRoughnessTexture']['index']]['uri'])
                    result = self.create_metallic_roughness_to_grayscale_images(metallic_roughness_texture_file)
                    metallic_factor = pbr_metallic_roughness['metallicFactor'] if 'metallicFactor' in pbr_metallic_roughness else 1.0
                    fallback_metallic = metallic_factor
                    scale_metallic = [metallic_factor] * 4
                    metallic_color_components = {
                        'b':
                        {'sdf_type' : Sdf.ValueTypeNames.Float, 'name': 'metallic'}
                    }

                    roughness_factor = pbr_metallic_roughness['roughnessFactor'] if 'roughnessFactor' in pbr_metallic_roughness else 1.0
                    fallback_roughness = roughness_factor
                    scale_roughness = [roughness_factor] * 4
                    roughness_color_components = {
                        'g':
                        {'sdf_type': Sdf.ValueTypeNames.Float, 'name': 'roughness'},
                    }


                    self._convert_texture_to_usd(
                        pbr_mat=pbr_mat,
                        gltf_texture=pbr_metallic_roughness['metallicRoughnessTexture'],
                        gltf_texture_name='metallicTexture',
                        color_components=metallic_color_components,
                        scale_factor=scale_metallic,
                        fallback_factor=fallback_metallic,
                        material_path=material_path,
                        fallback_type=Sdf.ValueTypeNames.Float,
                        primvar_st0_output=primvar_st0_output,
                        primvar_st1_output=primvar_st1_output
                    )

                    self._convert_texture_to_usd(
                        pbr_mat=pbr_mat,
                        gltf_texture=pbr_metallic_roughness['metallicRoughnessTexture'],
                        gltf_texture_name='roughnessTexture',
                        color_components=roughness_color_components,
                        scale_factor=scale_roughness,
                        fallback_factor=fallback_roughness,
                        material_path=material_path,
                        fallback_type=Sdf.ValueTypeNames.Float,
                        primvar_st0_output=primvar_st0_output,
                        primvar_st1_output=primvar_st1_output
                    )

                if pbr_specular_glossiness and 'specularGlossinessTexture' in pbr_specular_glossiness:
                    specular_glossiness_texture_file = os.path.join(self.gltf_loader.root_dir, self.gltf_loader.json_data['images'][pbr_specular_glossiness['specularGlossinessTexture']['index']]['uri'])
                    result = self.create_specular_glossiness_to_grayscale_images(specular_glossiness_texture_file)
                    specular_factor = tuple(pbr_specular_glossiness['specularFactor']) if 'specularFactor' in pbr_specular_glossiness else (1,1,1)
                    fallback_specular = specular_factor
                    scale_specular = [specular_factor[0], specular_factor[1], specular_factor[2], 1]
                    specular_color_components = {
                        'rgb':
                        {'sdf_type' : Sdf.ValueTypeNames.Color3f, 'name': 'specularColor'}
                    }

                    roughness_factor = 1 - pbr_specular_glossiness['glossiness'] if 'glossiness' in pbr_specular_glossiness else 0.0
                    fallback_roughness = roughness_factor
                    scale_roughness = [-1] * 4
                    glossiness_color_components = {
                        'a':
                        {'sdf_type': Sdf.ValueTypeNames.Float, 'name': 'roughness'},
                    }


                    self._convert_texture_to_usd(
                        pbr_mat=pbr_mat,
                        gltf_texture=pbr_specular_glossiness['specularGlossinessTexture'],
                        gltf_texture_name='specularTexture',
                        color_components=specular_color_components,
                        scale_factor=scale_specular,
                        fallback_factor=fallback_specular,
                        material_path=material_path,
                        fallback_type=Sdf.ValueTypeNames.Color3f,
                        primvar_st0_output=primvar_st0_output,
                        primvar_st1_output=primvar_st1_output
                    )

                    self._convert_texture_to_usd(
                        pbr_mat=pbr_mat,
                        gltf_texture=pbr_specular_glossiness['specularGlossinessTexture'],
                        gltf_texture_name='glossinessTexture',
                        color_components=glossiness_color_components,
                        scale_factor=[-1]*4,
                        fallback_factor=fallback_roughness,
                        material_path=material_path,
                        fallback_type=Sdf.ValueTypeNames.Float,
                        primvar_st0_output=primvar_st0_output,
                        primvar_st1_output=primvar_st1_output,
                        bias = (1.0, 1.0, 1.0, 1.0)
                    )

    def _convert_animations_to_usd(self):
        if hasattr(self, 'animations_map'):
            total_max_time = 0
            total_min_time = 0

            for node in self.animations_map:
                if node not in self.gltf_usdskel_nodemap:
                    channels = self.animations_map[node]
                    usd_node = self.gltf_usd_nodemap[node]

                    for channel in channels:
                        min_max_time = self._create_usd_animation(usd_node, channel)
                        total_max_time = max(total_max_time, min_max_time.max)
                        total_min_time = max(total_min_time, min_max_time.min)

            self.stage.SetStartTimeCode(total_min_time)
            self.stage.SetEndTimeCode(total_max_time)

        self._convert_skin_animations_to_usd()

    def _convert_to_usd_friendly_node_name(self, name):
        """Format a glTF name to make it more USD friendly

        Arguments:
            name {str} -- glTF node name

        Returns:
            str -- USD friendly name
        """
        #return name
        name = re.sub(r'\.|\b \b|-\b|:|\(|\)|[ \t]', '_', name) # replace '.',' ','-',':','/','\','(',')' and ':' with '_'
        return re.sub('//', '/', name)


    def _get_joint_name(self, joint_node):
        """Gets the joint name based on the glTF node hierarchy

        Arguments:
            joint_node {Node} -- glTF Node object

        Returns:
            str -- joint name
        """

        if len(joint_node.hierarchy_name) > 0:
            return joint_node.hierarchy_name[0]
        elif joint_node.parent == None:
            joint_node.hierarchy_name.append(joint_node.name)
        else:
            name = joint_node.name
            n = joint_node
            while(n.parent != None):
                n = self.node_hierarchy[n.parent]
                name = '{0}/{1}'.format(n.name, name)

            joint_node.hierarchy_name.append(name)

        return joint_node.hierarchy_name[0]

    def _get_gltf_root_joint_name(self, gltf_skin):
        if 'skeleton' in gltf_skin:
            node = self.node_hierarchy[gltf_skin['skeleton']]
            return self._convert_to_usd_friendly_node_name(node.name)

        else:
            if 'joints' in gltf_skin:
                joint_index = gltf_skin['joints'][0]
                joint_node = self.node_hierarchy[joint_index]

                while joint_node.parent != None:
                    joint_node = self.node_hierarchy[joint_node.parent]

                return self._convert_to_usd_friendly_node_name(joint_node.name)
            else:
                return None

    def _convert_skin_animations_to_usd(self):
        """Converts the skin animations to USD skeleton animations
        """
        if hasattr(self, 'animations_map'):
            total_max_time = 0
            total_min_time = 0

            if 'skins' in self.gltf_loader.json_data:
                for i, skin in enumerate(self.gltf_loader.json_data['skins']):
                    usd_mesh = self._usd_mesh_skin_map[i]
                    joint_map = {}
                    if 'joints' in skin:
                        joints = []
                        joint_anims = []
                        joint_values = []
                        for i, joint_index in enumerate(skin['joints']):
                            if joint_index in self.animations_map:
                                animation_channels = self.animations_map[joint_index]
                                skeleton_joint = self.gltf_usdskel_nodemap[joint_index]

                                joints.append(skeleton_joint)
                                joint_node = self.node_hierarchy[joint_index]
                                joint_name = self._convert_to_usd_friendly_node_name(self._get_joint_name(joint_node))

                                joint_data = JointData(skeleton_joint=skeleton_joint, joint_name=joint_name, joint_index=i)
                                joint_values.append(joint_data)

                                path_keyframes_map = {}
                                for animation_channel in animation_channels:
                                    if animation_channel.path not in joint_map:
                                        joint_map[animation_channel.path] = {}
                                    convert_func = self._get_keyframe_usdskel_value_conversion_func(animation_channel.path)
                                    path_keyframes_map[animation_channel.path] = []

                                    input_keyframe_accessor = self.gltf_loader.json_data['accessors'][animation_channel.sampler['input']]

                                    max_time = int(round(input_keyframe_accessor['max'][0] * self.fps))
                                    min_time = int(round(input_keyframe_accessor['min'][0] * self.fps))
                                    total_max_time = max(total_max_time, max_time)
                                    total_min_time = min(total_min_time, min_time)
                                    input_keyframes = self.gltf_loader.get_data(accessor=input_keyframe_accessor)
                                    output_keyframe_accessor = self.gltf_loader.json_data['accessors'][animation_channel.sampler['output']]
                                    output_keyframes = self.gltf_loader.get_data(accessor=output_keyframe_accessor)

                                    if len(input_keyframes) != len(output_keyframes):
                                        raise Exception('glTF animation input and output key frames must be the same length!')

                                    #loop through time samples
                                    for i, input_keyframe in enumerate(input_keyframes):
                                        frame = input_keyframe * self.fps
                                        value = convert_func(output_keyframes[i])
                                        if frame in joint_map[animation_channel.path]:
                                            joint_map[animation_channel.path][frame].append(value)
                                        else:
                                            joint_map[animation_channel.path][frame] = [value]

                                        keyframe = KeyFrame(input=frame, output=convert_func(output_keyframes[i]))
                                        path_keyframes_map[animation_channel.path].append(keyframe)

                                joint_anims.append(path_keyframes_map)

                        usd_skeleton = joints[0]['skeleton']
                        usd_animation = UsdSkel.Animation.Define(self.stage, '{0}/{1}'.format(usd_skeleton.GetPath(), 'anim'))

                        animated_joints = [x.joint_name for x in joint_values ]
                        usd_animation.CreateJointsAttr().Set(animated_joints)
                        self._store_joint_animations(usd_animation, joint_values, joint_map)

                        usd_skel_root_path = usd_skeleton.GetPath().GetParentPath()
                        usd_skel_root = self.stage.GetPrimAtPath(usd_skel_root_path)

                        UsdSkel.BindingAPI(usd_mesh).CreateAnimationSourceRel().AddTarget(usd_animation.GetPath())

            self.stage.SetStartTimeCode(total_min_time)
            self.stage.SetEndTimeCode(total_max_time)

    def _store_joint_animations(self, usd_animation, joint_data, joint_map):
        rotation_anim = usd_animation.CreateRotationsAttr()
        joint_count = len(joint_data)

        scale_anim = usd_animation.CreateScalesAttr()
        if 'scale' in joint_map:
            for joint in range(0, joint_count):
                scale_anim_data = joint_map['scale']
                for entry in scale_anim_data:
                    scale_anim.Set(time=entry, value=scale_anim_data[entry])
        else:
            rest_poses = []
            for joint in joint_data:
                scale = [1,1,1]
                skeleton_joint = joint.skeleton_joint['skeleton']
                joint_index = joint.joint_index
                joint_rest_transforms = skeleton_joint.GetRestTransformsAttr().Get()
                try:
                    scale[0] = joint_rest_transforms[joint_index].GetRow3(0).GetLength()
                    scale[1] = joint_rest_transforms[joint_index].GetRow3(1).GetLength()
                    scale[2] = joint_rest_transforms[joint_index].GetRow3(2).GetLength()
                except IndexError:
                    return
                rest_poses.append(scale)
            scale_anim.Set(rest_poses)

        if 'rotation' in joint_map:
            for i in range(0, joint_count):
                rotation_anim_data = joint_map['rotation']
                for entry in rotation_anim_data:
                    rotation_anim.Set(time=entry, value=rotation_anim_data[entry])
        else:
            rest_poses = []
            for joint in joint_data:
                rotation = joint.skeleton_joint['skeleton'].GetRestTransformsAttr().Get()[joint.joint_index].ExtractRotation().GetQuaternion()
                quat = Gf.Quatf()
                quat.SetReal(rotation.GetReal())
                quat.SetImaginary(Gf.Vec3f(rotation.GetImaginary()))
                rest_poses.append(quat)
            rotation_anim.Set(rest_poses)

        translation_anim = usd_animation.CreateTranslationsAttr()

        if 'translation' in joint_map:
            for i in range(0, joint_count):
                translation_anim_data = joint_map['translation']
                for entry in translation_anim_data:
                    translation_anim.Set(time=entry, value=translation_anim_data[entry])
        else:
            rest_poses = []
            for joint in joint_data:
                translation = joint.skeleton_joint['skeleton'].GetRestTransformsAttr().Get()[joint.joint_index].ExtractTranslation()
                rest_poses.append(translation)
            translation_anim.Set(rest_poses)

    def _convert_skin_to_usd(self, gltf_node, node_index, usd_parent_node, usd_mesh):
        """Converts a glTF skin to a UsdSkel

        Arguments:
            gltf_node {dict} -- glTF node
            node_index {int} -- index of the glTF node
            usd_parent_node {UsdPrim} -- parent node of the usd node
            usd_mesh {[type]} -- [description]
        """

        parent_path = usd_parent_node.GetPath()
        gltf_skin = self.gltf_loader.json_data['skins'][gltf_node['skin']]

        bind_matrices = []
        rest_matrices = []

        skeleton_root = self.stage.GetPrimAtPath(parent_path)
        skel_binding_api = UsdSkel.BindingAPI(usd_mesh)
        skel_binding_api_skel_root = UsdSkel.BindingAPI(usd_mesh)
        bind_matrices = self._compute_bind_transforms(gltf_skin)
        gltf_root_node_name = self._get_gltf_root_joint_name(gltf_skin)
        skeleton = UsdSkel.Skeleton.Define(self.stage, '{0}/{1}'.format(parent_path, gltf_root_node_name))
        skel_binding_api_skel_root.CreateSkeletonRel().AddTarget(skeleton.GetPath())
        if len(bind_matrices) > 0:
            skeleton.CreateBindTransformsAttr().Set(bind_matrices)

        joint_paths = []

        for joint_index in gltf_skin['joints']:
            joint_node = self.gltf_loader.json_data['nodes'][joint_index]

            rest_matrices.append(self._compute_rest_matrix(joint_node))

            node = self.node_hierarchy[joint_index]
            name = self._convert_to_usd_friendly_node_name(self._get_joint_name(node))

            joint_paths.append(Sdf.Path(name))

            self.gltf_usdskel_nodemap[joint_index] = {'skeleton':skeleton, 'joint_name': name}

        skeleton.CreateRestTransformsAttr().Set(rest_matrices)
        skeleton.CreateJointsAttr().Set(joint_paths)
        gltf_mesh = self.gltf_loader.json_data['meshes'][gltf_node['mesh']]
        if 'primitives' in gltf_mesh:
            primitive_attributes = gltf_mesh['primitives'][0]['attributes']
            if 'WEIGHTS_0' in primitive_attributes and 'JOINTS_0' in primitive_attributes:
                accessor = self.gltf_loader.json_data['accessors'][gltf_mesh['primitives'][0]['attributes']['WEIGHTS_0']]
                total_vertex_weights = self.gltf_loader.get_data(accessor)

                accessor = self.gltf_loader.json_data['accessors'][gltf_mesh['primitives'][0]['attributes']['JOINTS_0']]
                total_vertex_joints = self.gltf_loader.get_data(accessor)
                total_joint_indices = []
                total_joint_weights = []

                for joint_indices, weights in zip(total_vertex_joints, total_vertex_weights):
                    for joint_index, weight in zip(joint_indices, weights):
                        total_joint_indices.append(joint_index)
                        total_joint_weights.append(weight)

                joint_indices_attr = skel_binding_api.CreateJointIndicesPrimvar(False, 4).Set(total_joint_indices)
                total_joint_weights = Vt.FloatArray(total_joint_weights)
                UsdSkel.NormalizeWeights(total_joint_weights, 4)
                joint_weights_attr = skel_binding_api.CreateJointWeightsPrimvar(False, 4).Set(total_joint_weights)


    def _compute_bind_transforms(self, gltf_skin):
        """Compute the bind matrices from the skin

        Arguments:
            gltf_skin {dict} -- glTF skin

        Returns:
            [list] -- List of bind matrices
        """

        bind_matrices = []
        if 'inverseBindMatrices' in gltf_skin:
            inverse_bind_matrices_accessor = self.gltf_loader.json_data['accessors'][gltf_skin['inverseBindMatrices']]
            inverse_bind_matrices = self.gltf_loader.get_data(accessor=inverse_bind_matrices_accessor)

            for matrix in inverse_bind_matrices:
                bind_matrix = self._convert_to_usd_matrix(matrix)
                bind_matrices.append(bind_matrix.GetInverse())

        return bind_matrices

    def _convert_to_usd_matrix(self, matrix):
        """Converts a glTF matrix to a Usd Matrix

        Arguments:
            matrix {[type]} -- [description]

        Returns:
            [type] -- [description]
        """

        return Gf.Matrix4d(
            matrix[0], matrix[1], matrix[2], matrix[3],
            matrix[4], matrix[5], matrix[6], matrix[7],
            matrix[8], matrix[9], matrix[10], matrix[11],
            matrix[12], matrix[13], matrix[14], matrix[15]
        )

    def _compute_rest_matrix(self, gltf_node):
        """
        Compute the rest matrix from a glTF node.
        The translation, rotation and scale are combined into a transformation matrix

        Returns:
            Matrix4d -- USD matrix
        """

        xform_matrix = None
        if 'matrix' in gltf_node:
            matrix = gltf_node['matrix']
            xform_matrix = self._convert_to_usd_matrix(matrix)
            return xform_matrix
        else:
            usd_scale = Gf.Vec3h(1,1,1)
            usd_rotation = Gf.Quatf()
            usd_translation = Gf.Vec3f(0,0,0)
            if 'scale' in gltf_node:
                scale = gltf_node['scale']
                usd_scale = Gf.Vec3h(scale[0], scale[1], scale[2])

            if 'rotation' in gltf_node:
                rotation = gltf_node['rotation']
                usd_rotation = Gf.Quatf(rotation[3], rotation[0], rotation[1], rotation[2])

            if 'translation' in gltf_node:
                translation = gltf_node['translation']
                usd_translation = Gf.Vec3f(translation[0], translation[1], translation[2])

        return UsdSkel.MakeTransform(usd_translation, usd_rotation, usd_scale)

    def _build_node_hierarchy(self):
        """Constructs a node hierarchy from the glTF nodes

        Returns:
            dict -- dictionary of Node objects with node indices as the key
        """

        node_hierarchy = {}
        if 'nodes' in self.gltf_loader.json_data:
            for node_index, node in enumerate(self.gltf_loader.json_data['nodes']):
                new_node = None
                if node_index not in node_hierarchy:
                    node_name = self._convert_to_usd_friendly_node_name(node['name']) if 'name' in node else 'joint_{}'.format(node_index)
                    new_node = Node(index=node_index, parent=None, children=[], name=node_name.format(node_index), hierarchy_name=[])
                    node_hierarchy[node_index] = new_node
                else:
                    new_node = node_hierarchy[node_index]

                if 'children' in node:
                    for child_index in node['children']:
                        new_node.children.append(child_index)
                        if child_index not in node_hierarchy:
                            gltf_child_node = self.gltf_loader.json_data['nodes'][child_index]
                            child_node_name = self._convert_to_usd_friendly_node_name(gltf_child_node['name']) if 'name' in gltf_child_node else 'joint_{}'.format(child_index)
                            child_node = Node(index=child_index, parent=node_index, children=[], name=child_node_name, hierarchy_name=[])
                            node_hierarchy[child_index] = child_node

        return node_hierarchy



    def _create_usd_animation(self, usd_node, animation_channel):
        """Converts a glTF animation to a USD animation

        Arguments:
            usd_node {[type]} -- usd node
            animation_channel {AnimationMap} -- map of animation target path and animation sampler indices

        Returns:
            [type] -- [description]
        """

        sampler = animation_channel.sampler
        input_accessor = self.gltf_loader.json_data['accessors'][sampler['input']]
        max_time = int(round(input_accessor['max'][0] * self.fps))
        min_time = int(round(input_accessor['min'][0] * self.fps))
        input_keyframes = self.gltf_loader.get_data(accessor=input_accessor)
        output_accessor = self.gltf_loader.json_data['accessors'][sampler['output']]
        output_keyframes = self.gltf_loader.get_data(accessor=output_accessor)

        num_values = output_accessor['count'] / input_accessor['count']
        (transform, convert_func) = self._get_keyframe_conversion_func(usd_node, animation_channel)

        for i, keyframe in enumerate(input_keyframes):
            convert_func(transform, int(round(keyframe * self.fps)), output_keyframes, i, num_values)

        MinMaxTime = collections.namedtuple('MinMaxTime', ('max', 'min'))
        return MinMaxTime(max=max_time, min=min_time)

    def _get_keyframe_conversion_func(self, usd_node, animation_channel):
        """Convert glTF key frames to USD animation key frames

        Arguments:
            usd_node {UsdPrim} -- USD node to apply animations to
            animation_channel {obj} -- glTF animation

        Raises:
            Exception -- [description]

        Returns:
            [func] -- USD animation conversion function
        """

        path = animation_channel.path

        def convert_translation(transform, time, output, i, _):
            value = output[i]
            transform.Set(time=time, value=(value[0], value[1], value[2]))

        def convert_scale(transform, time, output, i, _):
            value = output[i]
            transform.Set(time=time, value=(value[0], value[1], value[2]))

        def convert_rotation(transform, time, output, i, _):
            value = output[i]
            transform.Set(time=time, value=Gf.Quatf(value[3], value[0], value[1], value[2]))

        def convert_weights(transform, time, output, i, values_per_step):
            start = i * values_per_step
            end = i * values_per_step + values_per_step
            values = output[i * values_per_step:i * values_per_step + values_per_step]
            value = list(map(lambda x: round(x, 5) + 0, values))
            transform.Set(time=time, value=value)

        if path == 'translation':
            return (usd_node.AddTranslateOp(opSuffix='translate'), convert_translation)
        elif path == 'rotation':
            return (usd_node.AddOrientOp(opSuffix='rotate'), convert_rotation)
        elif path == 'scale':
            return (usd_node.AddScaleOp(opSuffix='scale'), convert_scale)
        elif path == 'weights':
            prim = usd_node.GetPrim().GetChild("skeleton_root").GetChild("skel").GetChild("anim")
            anim_attr = prim.GetAttribute('blendShapeWeights')
            return (anim_attr, convert_weights)
        else:
            raise Exception('Unsupported animation target path! {}'.format(path))



    def _create_usd_skeleton_animation(self, usd_skeleton, sampler, gltf_target_path, joint_name, gltf_node):
        animation_map = self.animations_map[gltf_node]

        accessor = self.gltf_loader.json_data['accessors'][sampler['input']]
        max_time = int(round(accessor['max'][0] * self.fps))
        min_time = int(round(accessor['min'][0] * self.fps))
        input_keyframes = self.gltf_loader.get_data(accessor=accessor)
        accessor = self.gltf_loader.json_data['accessors'][sampler['output']]
        output_keyframes = self.gltf_loader.get_data(accessor=accessor)
        usd_animation = UsdSkel.Animation.Define(self.stage, '{0}/{1}'.format(usd_skeleton.GetPath(), 'anim'))
        usd_animation.CreateJointsAttr().Set([joint_name])
        usd_skel_root_path = usd_skeleton.GetPath().GetParentPath()
        usd_skel_root = self.stage.GetPrimAtPath(usd_skel_root_path)

        UsdSkel.BindingAPI(usd_skel_root).CreateAnimationSourceRel().AddTarget(usd_animation.GetPath())
        (transform, convert_func) = self._get_keyframe_usdskel_conversion_func(gltf_target_path, usd_animation)

        for i, keyframe in enumerate(input_keyframes):
            convert_func(transform, int(round(keyframe * self.fps)), output_keyframes[i])

        return (max_time, min_time)


    def _get_keyframe_usdskel_conversion_func(self, target_path, usd_animation):
        """Convert glTF keyframes to USD skeleton animations

        Arguments:
            target_path {str} -- glTF animation target path
            usd_animation {USD Animation} -- USD Animation

        Raises:
            Exception -- Unsupported animation path

        Returns:
            [func] -- USD animation conversion function
        """

        def convert_translation(transform, time, value):
            transform.Set(time=time, value=[(value[0], value[1], value[2])])

        def convert_scale(transform, time, value):
            transform.Set(time=time, value=[(value[0], value[1], value[2])])

        def convert_rotation(transform, time, value):
            transform.Set(time=time, value=[Gf.Quatf(value[3], value[0], value[1], value[2])])

        if target_path == 'translation':
            return (usd_animation.CreateTranslationsAttr(), convert_translation)
        elif target_path == 'rotation':
            return (usd_animation.CreateRotationsAttr(), convert_rotation)
        elif target_path == 'scale':
            return (usd_animation.CreateScalesAttr(), convert_scale)
        else:
            raise Exception('Unsupported animation target path! {}'.format(target_path))

    def _get_keyframe_usdskel_value_conversion_func(self, target_path):
        def convert_translation(value):
            return (value[0], value[1], value[2])

        def convert_scale(value):
            return (value[0], value[1], value[2])

        def convert_rotation(value):
            return Gf.Quatf(value[3], value[0], value[1], value[2])

        if target_path == 'translation':
            return convert_translation
        elif target_path == 'rotation':
            return convert_rotation
        elif target_path == 'scale':
            return convert_scale
        else:
            raise Exception('Unsupported animation target path! {}'.format(target_path))


    def unpack_textures_to_grayscale_images(self, image_path, color_components):
        image_base_name = ntpath.basename(image_path)
        texture_name = image_base_name
        for color_component, sdf_type in color_components.iteritems():
            if color_component == 'rgb':
                pass
            else:
                img = Image.open(image_path)
                if img.mode == 'P':
                    img = img.convert('RGB')
                if img.mode == 'RGB':
                    occlusion, roughness, metallic = img.split()
                    if color_component == 'r':
                        texture_name = 'Occlusion_{}'.format(image_base_name)
                        occlusion.save(os.path.join(self.output_dir, texture_name))
                    elif color_component == 'g':
                        texture_name = 'Roughness_{}'.format(image_base_name)
                        roughness.save(os.path.join(self.output_dir, texture_name))
                    elif color_component == 'b':
                        texture_name = 'Metallic_{}'.format(image_base_name)
                        metallic.save(os.path.join(self.output_dir, texture_name))
                elif img.mode == 'RGBA':
                    if color_component == 'a':
                        texture_name = 'Glossiness_{}'.format(image_base_name)
                        img.getchannel('A').save(os.path.join(self.output_dir, texture_name))
                    else:
                        raise Exception('unrecognized image type!: {}'.format(img.mode))
                elif img.mode == 'L':
                    #already single channel
                    pass
                else:
                    raise Exception('Unsupported image type!: {}'.format(img.mode))


        return texture_name

    def create_metallic_roughness_to_grayscale_images(self, image):
        image_base_name = ntpath.basename(image)
        roughness_texture_name = os.path.join(self.output_dir, 'Roughness_{}'.format(image_base_name))
        metallic_texture_name = os.path.join(self.output_dir, 'Metallic_{}'.format(image_base_name))

        img = Image.open(image)

        if img.mode == 'P':
            #convert paletted image to RGB
            img = img.convert('RGB')
        if img.mode == 'RGB':
            channels = img.split()
            #get roughness
            channels[1].save(roughness_texture_name)
            #get metalness
            channels[2].save(metallic_texture_name)

            return {'metallic': metallic_texture_name, 'roughness': roughness_texture_name}

    def create_specular_glossiness_to_grayscale_images(self, image):
        image_base_name = ntpath.basename(image)
        specular_texture_name = os.path.join(self.output_dir, 'Specular_{}'.format(image_base_name))
        glossiness_texture_name = os.path.join(self.output_dir, 'Glossiness_{}'.format(image_base_name))

        img = Image.open(image)

        if img.mode == 'P':
            #convert paletted image to RGBA
            img = img.convert('RGBA')
        if img.mode == 'RGBA':
            #get specular
            img.convert('RGB').save(specular_texture_name)
            #get glossiness
            img.getchannel('A').save(glossiness_texture_name)
            #channels[3].save(glossiness_texture_name)

            return {'specular': specular_texture_name, 'glossiness': glossiness_texture_name}

    '''
    Converts a glTF texture to USD
    '''
    def _convert_texture_to_usd(self, primvar_st0_output, primvar_st1_output, pbr_mat, gltf_texture, gltf_texture_name, color_components, scale_factor, fallback_factor, material_path, fallback_type, bias=None):
        image_name = gltf_texture if (isinstance(gltf_texture, basestring)) else self.images[gltf_texture['index']]
        image_path = os.path.join(self.output_dir, image_name)
        texture_index = int(gltf_texture['index'])
        texture = self.gltf_loader.json_data['textures'][texture_index]
        wrap_modes = self._get_texture__wrap_modes(texture)
        texture_shader = UsdShade.Shader.Define(self.stage, material_path.AppendChild(gltf_texture_name))
        texture_shader.CreateIdAttr("UsdUVTexture")

        texture_shader.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set(wrap_modes['wrapS'])
        texture_shader.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set(wrap_modes['wrapT'])
        if bias:
            texture_shader.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(bias)

        texture_name = self.unpack_textures_to_grayscale_images(image_path, color_components)
        file_asset = texture_shader.CreateInput('file', Sdf.ValueTypeNames.Asset)
        file_asset.Set(texture_name)

        for color_params, usd_color_params in color_components.iteritems():
            sdf_type = usd_color_params['sdf_type']
            texture_shader_output = texture_shader.CreateOutput(color_params, sdf_type)
            pbr_mat_texture = pbr_mat.CreateInput(usd_color_params['name'], sdf_type)
            pbr_mat_texture.ConnectToSource(texture_shader_output)

        texture_shader_input = texture_shader.CreateInput('st', Sdf.ValueTypeNames.Float2)
        texture_shader_fallback = texture_shader.CreateInput('fallback', fallback_type)
        texture_shader_fallback.Set(fallback_factor)
        if 'texCoord' in gltf_texture and gltf_texture['texCoord'] == 1:
            texture_shader_input.ConnectToSource(primvar_st1_output)
        else:
            texture_shader_input.ConnectToSource(primvar_st0_output)

        scale_vector = texture_shader.CreateInput('scale', Sdf.ValueTypeNames.Float4)
        scale_vector.Set((scale_factor[0], scale_factor[1], scale_factor[2], scale_factor[3]))

    def convert(self):
        if hasattr(self, 'gltf_loader'):
            self._init_animations_map()
            self._convert_images_to_usd()
            self._convert_materials_to_preview_surface()
            self.convert_nodes_to_xform()


def convert_to_usd(gltf_file, usd_file, fps, scale, verbose=False):
    """Converts a glTF file to USD

    Arguments:
        gltf_file {str} -- path to glTF file
        usd_file {str} -- path to write USD file

    Keyword Arguments:
        verbose {bool} -- [description] (default: {False})
    """

    GLTF2USD(gltf_file=gltf_file, usd_file=usd_file, fps=fps, scale=scale, verbose=verbose)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert glTF to USD')
    parser.add_argument('--gltf', '-g', action='store', dest='gltf_file', help='glTF file (in .gltf format)', required=True)
    parser.add_argument('--fps', action='store', dest='fps', help='The frames per second for the animations', type=float, default=24.0)
    parser.add_argument('--output', '-o', action='store', dest='usd_file', help='destination to store generated .usda file', required=True)
    parser.add_argument('--verbose', '-v', action='store_true', dest='verbose', help='Enable verbose mode')
    parser.add_argument('--scale', '-s', action='store', dest='scale', help='Scale the resulting USDA', type=float, default=100)
    args = parser.parse_args()

    if args.gltf_file:
        convert_to_usd(args.gltf_file, args.usd_file, args.fps, args.scale, args.verbose)
