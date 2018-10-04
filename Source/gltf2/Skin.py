from sets import Set
class Skin:
    """Represents a glTF Skin
    """

    def __init__(self, gltf2_loader, skin_entry):
        self._inverse_bind_matrices = self._init_inverse_bind_matrices(gltf2_loader, skin_entry)
        self._joints = [gltf2_loader.get_nodes()[joint_index] for joint_index in skin_entry['joints']]
        self._root_skeletons = self._init_root_skeletons(gltf2_loader, skin_entry)

    def _init_inverse_bind_matrices(self, gltf2_loader, skin_entry):
        inverse_bind_matrices = []
        if 'inverseBindMatrices' in skin_entry:
            inverse_bind_matrices = gltf2_loader.get_data(accessor=gltf2_loader.json_data['accessors'][skin_entry['inverseBindMatrices']])
        
        return inverse_bind_matrices

    def get_inverse_bind_matrices(self):
        return self._inverse_bind_matrices

    def get_joints(self):
        return self._joints

    def _init_root_skeletons(self, gltf2_loader, skin_entry):
        root_joints = Set()
        for joint_index in skin_entry['joints']:
            joint = gltf2_loader.nodes[joint_index]

            parent = joint.get_parent()
            while (parent != None):
                joint = parent
                parent = joint.get_parent()

            root_joints.add(joint)

        return list(root_joints)

    def get_bind_transforms(self):
        pass

    def get_rest_transforms(self):
        pass
    
    def get_joint_names(self):
        pass

    def get_root_joints(self):
        return self._root_skeletons

    def equals(self, gltf_skin):
        """Compares two skins to see if their values are equivalent to each other
        
        Arguments:
            gltf_skin Skin -- glTF skin
        
        Returns:
            [bool] -- Specifies if two skins are equivalent
        """

        joints = gltf_skin.get_joints()
        if len(set(joints)^set(self.get_joints())) != 0:
            return False
        if gltf_skin.get_inverse_bind_matrices() != self.get_inverse_bind_matrices():
            return False
        
        return True
