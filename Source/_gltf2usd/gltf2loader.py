import codecs
from enum import Enum
import base64
import json
import os
import re
import struct

import gltf2usdUtils

from gltf2 import Skin, Node, Animation, Scene, Mesh, Material, GLTFImage



class AccessorType(Enum):
    SCALAR = 'SCALAR'
    VEC2 = 'VEC2'
    VEC3 = 'VEC3'
    VEC4 = 'VEC4'
    MAT2 = 'MAT2'
    MAT3 = 'MAT3'
    MAT4 = 'MAT4'

class AccessorComponentType(Enum):
    BYTE = 5120
    UNSIGNED_BYTE = 5121
    SHORT = 5122
    UNSIGNED_SHORT = 5123
    UNSIGNED_INT = 5125
    FLOAT = 5126

class TextureWrap(Enum):
    CLAMP_TO_EDGE = 33071
    MIRRORED_REPEAT = 33648
    REPEAT = 10497

class MagFilter(Enum):
    NEAREST = 9728
    LINEAR = 9729

class MinFilter(Enum):
    NEAREST = 9728
    LINEAR = 9729
    NEAREST_MIPMAP_NEAREST = 9984
    LINEAR_MIPMAP_NEAREST = 9985
    NEAREST_MIPMAP_LINEAR = 9986
    LINEAR_MIPMAP_LINEAR = 9987

class AccessorTypeCount(Enum):
    SCALAR = 1
    VEC2 = 2
    VEC3 = 3
    VEC4 = 4
    MAT2 = 4
    MAT3 = 9
    MAT4 = 16

def accessor_type_count(x):
    return {
        'SCALAR': 1,
        'VEC2': 2,
        'VEC3': 3,
        'VEC4': 4,
        'MAT2': 4,
        'MAT3': 9,
        'MAT4': 16
    }[x]

def PrimitiveMode(Enum):
    POINTS = 0
    LINES = 1
    LINE_LOOP = 2
    LINE_STRIP = 3
    TRIANGLES = 4
    TRIANGLE_STRIP = 5
    TRIANGLE_FAN = 6

def accessor_component_type_bytesize(x):
    return {
        AccessorComponentType.BYTE: 1,
        AccessorComponentType.UNSIGNED_BYTE: 1,
        AccessorComponentType.SHORT: 2,
        AccessorComponentType.UNSIGNED_SHORT: 2,
        AccessorComponentType.UNSIGNED_INT: 4,
        AccessorComponentType.FLOAT: 4,
    }[x]




class GLTF2Loader:
    """A very simple glTF loader.  It is essentially a utility to load data from accessors
    """

    def __init__(self, gltf_file, optimize_textures=False):
        """Initializes the glTF 2.0 loader

        Arguments:
            gltf_file {str} -- Path to glTF file
        """
        if not os.path.isfile(gltf_file):
            raise Exception("file {} does not exist".format(gltf_file))

        if not gltf_file.endswith('.gltf'):
            raise Exception('Can only accept .gltf files')

        self.root_dir = os.path.dirname(gltf_file)
        self._optimize_textures = optimize_textures
        try:
            with codecs.open(gltf_file, encoding='utf-8', errors='strict') as f:
                self.json_data = json.load(f)
        except UnicodeDecodeError:
            with open(gltf_file) as f:
                self.json_data = json.load(f)

        self._initialize()

    def _initialize(self):
        """Initializes the glTF loader
        """
        self._initialize_images()
        self._initialize_materials()
        self._initialize_meshes()
        self._initialize_nodes()
        self._initialize_skins()
        self._initialize_scenes()
        
        self._initialize_animations()

    def _initialize_images(self):
        self._images = []
        if 'images' in self.json_data:
            for i, image_entry in enumerate(self.json_data['images']):
                self._images.append(GLTFImage.GLTFImage(image_entry, i, self, self._optimize_textures))


    def _initialize_nodes(self):
        self.nodes = []
        if 'nodes' in self.json_data:
            for i, node_entry in enumerate(self.json_data['nodes']):
                node = Node(node_entry, i, self)
                self.nodes.append(node)

            for i, node_entry in enumerate(self.json_data['nodes']):
                if 'children' in node_entry:
                    parent = self.nodes[i]
                    for child_index in node_entry['children']:
                        child = self.nodes[child_index]
                        child._parent = parent
                        parent._children.append(child)

    def _initialize_materials(self):
        self._materials = []

        if 'materials' in self.json_data:
            for i, material_entry in enumerate(self.json_data['materials']):
                material = Material(material_entry, i, self)
                self._materials.append(material)

    def _initialize_scenes(self):
        self._scenes = []
        self._main_scene = None
        if 'scenes' in self.json_data:
            for i, scene_entry in enumerate(self.json_data['scenes']):
                scene = Scene(scene_entry, i, self.nodes)
                self._scenes.append(scene)

            if 'scene' in self.json_data:
                self._main_scene = self._scenes[self.json_data['scene']]
            else:
                self._main_scene = self._scenes[0]

    def get_images(self):
        return self._images

    def get_scenes(self):
        """Get the scene objects from the glTF file
        
        Returns:
            Scene[] -- glTF scene objects
        """

        return self._scenes

    def get_main_scene(self):
        """Returns the main scene in the glTF file, or none if there are no scenes
        
        Returns:
            Scene -- glTF scene
        """

        return self._main_scene

    def get_materials(self):
        return self._materials

    def get_meshes(self):
        return self._meshes

    def _initialize_meshes(self):
        self._meshes = []
        if 'meshes' in self.json_data:
            for i, mesh_entry in enumerate(self.json_data['meshes']):
                mesh = Mesh(mesh_entry, i, self)
                self._meshes.append(mesh)



    def _initialize_animations(self):
        self.animations = []
        if 'animations' in self.json_data:
            for i, animation_entry in enumerate(self.json_data['animations']):
                animation = Animation(animation_entry, i, self)
                self.animations.append(animation)


    def _initialize_skins(self):
        self.skins = []
        if 'skins' in self.json_data:
            self.skins = [Skin(self, skin) for skin in self.json_data['skins']]
            for node in self.nodes:
                if node._skin_index != None:
                    node._skin = self.skins[node._skin_index]

    def get_nodes(self):
        return self.nodes

    def get_skins(self):
        return self.skins

    def get_animations(self):
        return self.animations


    def align(self, value, size):
        remainder = value % size
        return value if (remainder == 0) else (value + size - remainder)

    def get_data(self, accessor):
        bufferview = self.json_data['bufferViews'][accessor['bufferView']]
        buffer = self.json_data['buffers'][bufferview['buffer']]
        accessor_type = AccessorType(accessor['type'])
        uri = buffer['uri']
        buffer_data = ''

        if re.match(r'^data:.*?;base64,', uri):
            uri_data = uri.split(',')[1]
            buffer_data = base64.b64decode(uri_data)
            if 'byteOffset' in bufferview:
                buffer_data = buffer_data[bufferview['byteOffset']:]
        else:
            buffer_file = os.path.join(self.root_dir, uri)
            with open(buffer_file, 'rb') as buffer_fptr:
                if 'byteOffset' in bufferview:
                    buffer_fptr.seek(bufferview['byteOffset'], 1)

                buffer_data = buffer_fptr.read(bufferview['byteLength'])

        data_arr = []
        accessor_component_type = AccessorComponentType(accessor['componentType'])

        accessor_type_size = accessor_type_count(accessor['type'])
        accessor_component_type_size = accessor_component_type_bytesize(accessor_component_type)

        bytestride = int(bufferview['byteStride']) if ('byteStride' in bufferview) else (accessor_type_size * accessor_component_type_size)
        offset = int(accessor['byteOffset']) if 'byteOffset' in accessor else 0

        data_type = ''
        data_type_size = 4
        normalize_divisor = 1.0 #used if the value needs to be normalized
        if accessor_component_type == AccessorComponentType.FLOAT:
            data_type = 'f'
            data_type_size = 4
        elif accessor_component_type == AccessorComponentType.UNSIGNED_INT:
            data_type = 'I'
            data_type_size = 4
        elif accessor_component_type == AccessorComponentType.UNSIGNED_SHORT:
            data_type = 'H'
            data_type_size = 2
            normalize_divisor = 65535.0 if 'normalized' in accessor and accessor['normalized'] == True else 1.0 
        elif accessor_component_type == AccessorComponentType.UNSIGNED_BYTE:
            data_type = 'B'
            data_type_size = 1
            normalize_divisor = 255.0 if 'normalized' in accessor and accessor['normalized'] == True else 1.0
        else:
            raise Exception('unsupported accessor component type!')

        for i in range(0, accessor['count']):
            entries = []
            for j in range(0, accessor_type_size):
                x = offset + j * accessor_component_type_size
                window = buffer_data[x:x + data_type_size]
                entries.append((struct.unpack(data_type, window)[0])/normalize_divisor)

            if len(entries) > 1:
                data_arr.append(tuple(entries))
            else:
                data_arr.append(entries[0])
            offset = offset + bytestride

        return data_arr
