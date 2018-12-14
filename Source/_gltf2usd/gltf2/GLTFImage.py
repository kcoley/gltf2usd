import base64
from io import BytesIO
import math
import ntpath
import os

from enum import Enum
from PIL import Image
import numpy as np

class ImageColorChannels(Enum):
    RGB = 'RGB'
    RGBA = 'RGBA'
    R = 'R'
    G = 'G'
    B = 'B'
    A = 'A'

class GLTFImage(object):
    def __init__(self, image_entry, image_index, gltf_loader, optimize_textures=False):
        self._optimize_textures = optimize_textures

        if image_entry['uri'].startswith('data:image'):
            uri_data = image_entry['uri'].split(',')[1]
            img = Image.open(BytesIO(base64.b64decode(uri_data)))

            # NOTE: image might not have a name
            self._name = image_entry['name'] if 'name' in image_entry else 'image_{}.{}'.format(image_index, img.format.lower())
            self._image_path = os.path.join(gltf_loader.root_dir, self._name)
            img.save(self._image_path, optimize=self._optimize_textures)
        else:
            self._uri = image_entry['uri']
            self._name = ntpath.basename(self._uri)
            self._image_path = os.path.join(gltf_loader.root_dir, self._uri)

        #decode unicode name to ascii
        if isinstance(self._name, unicode):
            self._name = self._name.encode('utf-8')
            self._name = self._name.decode('ascii', 'ignore')

    def get_image_path(self):
        return self._image_path

    def write_to_directory(self, output_dir, channels, texture_prefix, offset = [0,0], scale = [1,1], rotation = 0):
        file_name = '{0}_{1}'.format(texture_prefix, ntpath.basename(self._name)) if texture_prefix else ntpath.basename(self._name)
        destination = os.path.join(output_dir, file_name)
        original_img = Image.open(self._image_path)
        img = original_img
        if img.mode == 'P':
            img = img.convert('RGBA')
        img_channels = img.split()
        if len(img_channels) == 1: #distribute grayscale image across rgb
            img = Image.merge('RGB', (img_channels[0], img_channels[0], img_channels[0]))
            img_channels = img.split()
        if channels == ImageColorChannels.RGB:
            if img.mode == "RGBA": #Make a copy and add opaque 
                file_name = '{0}_{1}'.format('RGB', file_name)
                destination = os.path.join(output_dir, file_name)

            img = Image.merge('RGB', (img_channels[0], img_channels[1], img_channels[2]))
        elif channels == ImageColorChannels.RGBA:
            img = original_img.convert('RGBA')
        elif channels == ImageColorChannels.R or channels == ImageColorChannels.G or channels == ImageColorChannels.B or channels == ImageColorChannels.A:
            if img.mode != 'L':
                img = img.getchannel(channels.value)

        else:
            raise Exception('Unsupported image channel format {}'.format(channels))

        if destination.endswith('jpg') or destination.endswith('.jpeg'):
            img = img.convert('RGB')

        #apply texture transform
        texture_transform_matrix = self._texture_transform_matrix(offset, scale, rotation)
        img = img.transform((img.width, img.height), Image.AFFINE, (
            texture_transform_matrix[0,0], texture_transform_matrix[0,1], texture_transform_matrix[0,2],
            texture_transform_matrix[1,0], texture_transform_matrix[1,1], texture_transform_matrix[1,2]
            )
        )
        
        
        
        img.save(destination, optimize=self._optimize_textures)
        
        return file_name 

    def _texture_transform_matrix(self, offset, scale, rotation):
        translation_matrix = np.matrix([[1,0,offset[0]], [0,1,offset[1]], [0, 0, 1]])
        rotation_matrix = np.matrix([[math.cos(rotation), math.sin(rotation), 0], [-math.sin(rotation), math.cos(rotation), 0], [0,0,1]])
        scale_matrix = np.matrix(
            [
                [scale[0], 0, 0], 
                [0, scale[1], 0], 
                [0, 0, 1]
            ]
        )
        transform_matrix = np.matmul(np.matmul(translation_matrix, rotation_matrix), scale_matrix)

        return transform_matrix


