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
        if 'bufferView' in image_entry:
            #get image data from bufferview
            bufferview = gltf_loader.json_data['bufferViews'][image_entry['bufferView']]
            if 'byteOffset' in bufferview:
                buffer = gltf_loader.json_data['buffers'][bufferview['buffer']]

                img_base64 = buffer['uri'].split(',')[1]
                buff = BytesIO()
                buff.write(base64.b64decode(img_base64))
                buff.seek(bufferview['byteOffset'])
                img = Image.open(BytesIO(buff.read(bufferview['byteLength'])))
                # NOTE: image might not have a name
                self._name = image_entry['name'] if 'name' in image_entry else 'image_{}.{}'.format(image_index, img.format.lower())
                self._image_path = os.path.join(gltf_loader.root_dir, self._name)
                img.save(self._image_path, optimize=self._optimize_textures)
        else:
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

        #apply texture transform if necessary
        if offset != [0,0] or scale != [1,1] or rotation != 0:
            texture_transform_prefix_name= 'o{0}{1}s{2}{3}r{4}_'.format(offset[0], offset[1], scale[0], scale[1], rotation).replace('.', '_')
            file_name = texture_transform_prefix_name + file_name
            destination = os.path.join(output_dir, file_name)
            print('Generating texture transformed image "{}" ...'.format(file_name))
            img = self._transform_image(img, translate=offset, scale=scale, rotation=rotation)

        img.save(destination, optimize=self._optimize_textures)
        
        return file_name 

    def _texture_transform_matrix(self, offset, scale, rotation):
        """Creates a texture transform matrix, based on specification for KHR_texture_transform (https://github.com/KhronosGroup/glTF/tree/master/extensions/2.0/Khronos/KHR_texture_transform)
        
        Arguments:
            offset {list} -- The offset of the UV coordinate origin as a factor of the texture dimensions
            scale {list} -- The scale factor applied to the components of the UV coordinates
            rotation {float} -- Rotate the UVs by this many radians counter-clockwise around the origin
        
        Returns:
            [numpy.matrix] -- texture transform matrix
        """

        pivot_center = [0.0, -1.0]
        rotation *= -1
        pre_translation_matrix = np.matrix([[1, 0, -pivot_center[0]], [0, 1, -pivot_center[1]], [0, 0, 1]])
        post_translation_matrix = np.matrix([[1, 0, pivot_center[0]], [0, 1, pivot_center[1]], [0, 0, 1]])
        translation_matrix = np.matrix(
            [
                [1,0,offset[0]], 
                [0,1,offset[1]], 
                [0, 0, 1]
            ]
        )
        rotation_matrix = np.matrix(
            [
                [math.cos(rotation), math.sin(rotation), 0], 
                [-math.sin(rotation), math.cos(rotation), 0], 
                [0, 0, 1]
            ]
        )
        scale_matrix = np.matrix(
            [
                [scale[0], 0, 0], 
                [0, scale[1], 0], 
                [0, 0, 1]
            ]
        )
        transform_matrix = np.matmul(np.matmul(pre_translation_matrix, np.matmul(np.matmul(translation_matrix, rotation_matrix), scale_matrix)), post_translation_matrix)
 
        return transform_matrix

    def _transform_image(self, img, scale, rotation, translate):
        """Generates a new texture transformed image
        
        Arguments:
            img {Image} -- source image
            scale {list} -- scale of the new image
            rotation {float} -- rotation of the new image
            translate {list} -- translation of the new image
        
        Returns:
            [Image] -- transformed image
        """

        def _normalized_texcoord(x):
            return  x - int(x) if x >= 0 else 1 + (x - int(x))
            
        texture_transform_matrix = self._texture_transform_matrix(translate, scale, rotation)
        width = img.width
        height = img.height

        res = np.matmul(texture_transform_matrix, np.array([0.0,0.0,1]))

        source_image_pixels = img.getdata()
        new_img = Image.new(img.mode, (img.width, img.height))
        
        pixels = new_img.load()

        for col in range(new_img.size[0]):
            for row in range(new_img.size[1]):
                res = np.matmul(texture_transform_matrix, np.array([col/float(img.width),row/float(img.height),1]))

                c = min(int(round(_normalized_texcoord(res[0,0]) * height)), img.height - 1)
                r = min(int(round(_normalized_texcoord(res[0,1]) * width)), img.width - 1)
                pixel = source_image_pixels[r * width + c]
                pixels[col, row] = pixel

        return new_img


