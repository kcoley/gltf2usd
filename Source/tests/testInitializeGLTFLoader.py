import unittest

import sys
import os
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from _gltf2usd.gltf2loader import GLTF2Loader
from _gltf2usd.gltf2usdUtils import GLTF2USDUtils

class TestInitializeGLTFLoader(unittest.TestCase):
    def setUp(self):
        gltf_file = os.path.join("tests", "assets", "Start_Walking", "Start_Walking.gltf")
        self.loader = GLTF2Loader(gltf_file)

    def test_get_nodes(self):
        nodes = self.loader.get_nodes()
        for node in nodes:
            print(node.parent)

    def test_get_scenes(self):
        scenes = self.loader.get_scenes()

    def test_get_main_scene(self):
        main_scene = self.loader.get_main_scene()

    def test_get_nodes_from_scene(self):
        main_scene = self.loader.get_main_scene()
        nodes = main_scene.get_nodes()

    def test_get_mesh(self):
        node = [node for node in self.loader.get_nodes() if node.get_mesh() != None][0]
        mesh = node.get_mesh()

    def test_get_mesh_primitive(self):
        node = [node for node in self.loader.get_nodes() if node.get_mesh() != None][0]
        mesh = node.get_mesh()
        primitive = mesh.get_primitives()
        

    def test_get_skins(self):
        skins = self.loader.get_skins()
        for skin in skins:
            print(skin)

    def test_get_skin_inverse_bind_matrices(self):
        skin = self.loader.get_skins()[0]
        inverse_bind_matrices = skin.get_inverse_bind_matrices()
        for ibm in inverse_bind_matrices:
            print(ibm)

    def test_get_skin_joints(self):
        skin = self.loader.get_skins()[0]
        joints = skin.get_joints()

        for joint in joints:
            print(joint.name)

    def test_get_root_skin_joint(self):
        skin = self.loader.get_skins()[0]
        root_joints = skin.root_joints

    def test_skin_get_inverse_bind_matrices(self):
        skin = self.loader.get_skins()[0]
        skin.get_inverse_bind_matrices()

    def test_convert_node_name_to_usd_friendly_name(self):
        node = self.loader.get_nodes()[0]
        GLTF2USDUtils.convert_to_usd_friendly_node_name(node.name)

    def test_convert_node_transform_to_rest_matrix(self):
        node = self.loader.get_nodes()[0]
        GLTF2USDUtils.compute_usd_transform_matrix_from_gltf_node(node)

    def test_get_animation_channels_for_node_index_and_path(self):
        animation = self.loader.get_animations()[0]
        node = self.loader.get_nodes()[2]
        animation_channel = animation.get_animation_channel_for_node_and_path(node, 'rotation')

    def test_get_animation_channels_for_node_index(self):
        animation = self.loader.get_animations()[0]
        node = self.loader.get_nodes()[2]
        animation_channels = animation.get_animation_channels_for_node(node)
        print('\n\n\n\n\n\n\n')
        print([animation_channel.target.path for animation_channel in animation_channels])

    def test_get_animation_sampler_input_data(self):
        animation = self.loader.get_animations()[0]
        node = self.loader.get_nodes()[2]
        animation_channel = animation.get_animation_channel_for_node_and_path(node, 'rotation')
        input_data = animation_channel.sampler.get_input_data()

        print('input:\n\n\n\n\n\n\n')
        print(input_data)

    def test_get_animation_sampler_output_data(self):
        animation = self.loader.get_animations()[0]
        node = self.loader.get_nodes()[2]
        animation_channel = animation.get_animation_channel_for_node_and_path(node, 'rotation')
        output_data = animation_channel.sampler.get_output_data()
        print('output:\n\n\n\n\n\n\n')
        print(output_data)


    

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestInitializeGLTFLoader)
    unittest.TextTestRunner(verbosity=2).run(suite)