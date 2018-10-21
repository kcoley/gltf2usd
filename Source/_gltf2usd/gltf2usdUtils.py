import re

from pxr import Gf, UsdSkel

class GLTF2USDUtils:
    @staticmethod
    def convert_to_usd_friendly_node_name(name):
        """Format a glTF name to make it more USD friendly

        Arguments:
            name {str} -- glTF node name

        Returns:
            str -- USD friendly name
        """
        name = re.sub(r'\.|\b \b|-\b|:|\(|\)|[ \t]|-', '_', name) # replace '.',' ','-',':','/','\','(',')' and ':' with '_'
        name = re.sub(r'^([\d]+)',r'm\1',name)
        return re.sub('//', '/', name)

    @staticmethod
    def get_skin_rest_transforms(gltf_skin):
        joints = gltf_skin.get_joints()
        for joint in joints:
            rest_matrix = GLTF2USDUtils.compute_usd_transform_matrix_from_gltf_node(joint)

    @staticmethod
    def compute_usd_transform_matrix_from_gltf_node(node):
        """Computes a transform matrix from a glTF node object
        
        Arguments:
            node {Node} -- glTF2 Node object
        
        Returns:
            Matrix-- USD matrix
        """

        matrix = node.matrix
        if (matrix != None):
            return Gf.Matrix4d(
                matrix[0], matrix[1], matrix[2], matrix[3],
                matrix[4], matrix[5], matrix[6], matrix[7],
                matrix[8], matrix[9], matrix[10], matrix[11],
                matrix[12], matrix[13], matrix[14], matrix[15]
            )
        else:
            translation = node.translation
            usd_translation = Gf.Vec3f(translation[0], translation[1], translation[2])

            rotation = node.rotation
            usd_rotation = Gf.Quatf(rotation[3], rotation[0], rotation[1], rotation[2])

            scale = node.scale
            usd_scale = Gf.Vec3h(scale[0], scale[1], scale[2])

        return UsdSkel.MakeTransform(usd_translation, usd_rotation, usd_scale)

    @staticmethod
    def slerp(vec0, vec1, factor):
        quat0 = Gf.Quatf(vec0[3], vec0[0], vec0[1], vec0[2])
        quat1 = Gf.Quatf(vec1[3], vec1[0], vec1[1], vec1[2])
        result = Gf.Quatf(Gf.Slerp(factor, quat0, quat1))

        return result
