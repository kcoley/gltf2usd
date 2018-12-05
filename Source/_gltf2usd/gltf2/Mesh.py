from enum import Enum

class PrimitiveModeType(Enum):
    POINTS = 0
    LINES = 1
    LINE_LOOP = 2
    LINE_STRIP = 3
    TRIANGLES = 4
    TRIANGLE_STRIP = 5
    TRIANGLE_FAN = 6

class PrimitiveAttribute:
    def __init__(self, attribute_name, attribute_data, accessor_type, min_value=None, max_value=None):
        self._attribute_type = attribute_name
        self._attribute_data = attribute_data
        self._accessor_type = accessor_type
        self._min_value = min_value
        self._max_value = max_value

    @property
    def attribute_type(self):
        return self._attribute_type

    @property
    def accessor_type(self):
        return self._accessor_type

    def get_min_value(self):
        return self._min_value

    def get_max_value(self):
        return self._max_value

    def get_data(self):
        return self._attribute_data


class PrimitiveTarget:
    def __init__(self, target_entry, target_index, gltf_loader):
        self._name = None
        self._attributes = {}
        for entry in target_entry:
            accessor = gltf_loader.json_data['accessors'][target_entry[entry]]
            if self._name == None:
                self._name = accessor['name'] if ('name' in accessor) else 'shape_{}'.format(target_index)
            data = gltf_loader.get_data(accessor)
            self._attributes[entry] = data

    def get_attributes(self):
        return self._attributes

    def get_name(self):
        return self._name


class Primitive:
    def __init__(self, primitive_entry, i, gltf_mesh, gltf_loader):
        self._name = primitive_entry['name'] if ('name' in primitive_entry) else 'primitive_{}'.format(i)
        self._attributes = {}
        self._material = None
        self._targets = []

        if 'attributes' in primitive_entry:
            for attribute_name in primitive_entry['attributes']:
                accessor_index = primitive_entry['attributes'][attribute_name]
                accessor = gltf_loader.json_data['accessors'][accessor_index]
                data = gltf_loader.get_data(accessor)
                min_value = accessor['min'] if ('min' in accessor) else None
                max_value = accessor['max'] if ('max' in accessor) else None

                self._attributes[attribute_name] = PrimitiveAttribute(attribute_name, data, accessor['type'], min_value, max_value)

        self._indices = self._get_indices(primitive_entry, gltf_loader)
        self._mode = PrimitiveModeType(primitive_entry['mode']) if ('mode' in primitive_entry) else PrimitiveModeType.TRIANGLES
        if 'material' in primitive_entry:
            self._material = gltf_loader.get_materials()[primitive_entry['material']]

        # Fetch the names and accessors of the blendshapes
        if 'targets' in primitive_entry:
            for i, target_entry in enumerate(primitive_entry['targets']):
                target = PrimitiveTarget(target_entry, i, gltf_loader)
                self._targets.append(target)

    def _get_indices(self, primitive_entry, gltf_loader):
        if 'indices' in primitive_entry:
            accessor_index = primitive_entry['indices']
            accessor = gltf_loader.json_data['accessors'][accessor_index]
            data = gltf_loader.get_data(accessor)
            return data

        else:
            position_accessor = gltf_loader.json_data['accessors'][primitive_entry['attributes']['POSITION']]
            count = position_accessor['count']
            return range(0, count)

    def get_attributes(self):
        return self._attributes

    def get_morph_targets(self):
        return self._targets

    def get_indices(self):
        return self._indices

    def get_material(self):
        return self._material

    def get_name(self):
        return self._name



class Mesh:
    def __init__(self, mesh_entry, mesh_index, gltf_loader):
        self._primitives = []
        self._weights = []
        self._index = mesh_index
        if 'weights' in mesh_entry:
            self._weights = mesh_entry['weights']
        if 'primitives' in mesh_entry:
            for i, primitive_entry in enumerate(mesh_entry['primitives']):
                primitive = Primitive(primitive_entry, i, self, gltf_loader)
                self._primitives.append(primitive)

    def get_weights(self):
        return self._weights

    def get_primitives(self):
        return self._primitives


