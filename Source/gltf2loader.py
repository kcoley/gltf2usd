from enum import Enum
import json
import os
import struct
import base64

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

    def __init__(self, gltf_file):
        """Initializes the glTF 2.0 loader

        Arguments:
            gltf_file {str} -- Path to glTF file
        """
        if not os.path.isfile(gltf_file):
            raise Exception("file {} does not exist".format(gltf_file))

        if not gltf_file.endswith('.gltf'):
            raise Exception('Can only accept .gltf files')

        self.root_dir = os.path.dirname(gltf_file)
        with open(gltf_file) as f:
            self.json_data = json.load(f)


    def align(self, value, size):
        remainder = value % size
        return value if (remainder == 0) else (value + size - remainder)

    def get_data(self, accessor):
        bufferview = self.json_data['bufferViews'][accessor['bufferView']]
        buffer = self.json_data['buffers'][bufferview['buffer']]
        accessor_type = AccessorType(accessor['type'])


        if uri.startswith('data:application/octet-stream;base64,'):
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
        if accessor_component_type == AccessorComponentType.FLOAT:
            data_type = 'f'
            data_type_size = 4
        elif accessor_component_type == AccessorComponentType.UNSIGNED_INT:
            data_type = 'I'
            data_type_size = 4
        elif accessor_component_type == AccessorComponentType.UNSIGNED_SHORT:
            data_type = 'H'
            data_type_size = 2
        elif accessor_component_type == AccessorComponentType.UNSIGNED_BYTE:
            data_type = 'B'
            data_type_size = 1
        else:
            raise Exception('unsupported accessor component type!')
        
        for i in range(0, accessor['count']):
            entries = []
            for j in range(0, accessor_type_size):
                x = offset + j * accessor_component_type_size
                window = buffer_data[x:x + data_type_size]
                entries.append(struct.unpack(data_type, window)[0])
            if len(entries) > 1:
                data_arr.append(tuple(entries))
            else:
                data_arr.append(entries[0])
            offset = offset + bytestride

        return data_arr
