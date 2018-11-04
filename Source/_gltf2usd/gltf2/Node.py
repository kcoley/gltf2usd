import unicodedata

class Node:
    """Create a glTF node object
    """

    def __init__(self, node_dict, node_index, gltf_loader):
        self._parent = None
        self._name = node_dict['name'] if ('name' in node_dict and len(node_dict['name']) > 0) else 'node_{}'.format(node_index)
        
        if isinstance(self._name, unicode):
            self._name = unicodedata.normalize('NFKD', self._name).encode('ascii', 'ignore')

        self._name = '{0}_{1}'.format(self._name, node_index)
        self._node_index = node_index
        self._matrix = node_dict['matrix'] if ('matrix' in node_dict) else None
        self._translation = node_dict['translation'] if ('translation' in node_dict) else [0,0,0]
        self._rotation = node_dict['rotation'] if ('rotation' in node_dict) else [0,0,0,1]     
        self._scale = node_dict['scale'] if ('scale' in node_dict) else [1,1,1]   
        self._skin_index = node_dict['skin'] if ('skin' in node_dict) else None
        self._skin = None
        
        self._mesh = gltf_loader.get_meshes()[node_dict['mesh']] if ('mesh' in node_dict) else None
        
        self._children_indices = node_dict['children'] if ('children' in node_dict) else []
        self._children = []

    @property
    def name(self):
        return self._name

    @property
    def translation(self):
        return self._translation

    @property
    def rotation(self):
        return self._rotation

    @property
    def scale(self):
        return self._scale

    def get_children(self):
        return self._children

    @property
    def parent(self):
        return self._parent

    @property
    def matrix(self):
        return self._matrix

    def get_index(self):
        return self._node_index

    def get_mesh(self):
        return self._mesh

    def get_skin(self):
        return self._skin

    @property
    def index(self):
        return self._node_index