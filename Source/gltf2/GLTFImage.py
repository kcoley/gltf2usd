import base64
from io import BytesIO
import ntpath
import os

from enum import Enum
from PIL import Image

class ImageColorChannels(Enum):
    RGB = 'RGB'
    RGBA = 'RGBA'
    R = 'R'
    G = 'G'
    B = 'B'
    A = 'A'

class GLTFImage(object):
    def __init__(self, image_entry, image_index, gltf_loader):
        if image_entry['uri'].startswith('data:image'):
            uri_data = image_entry['uri'].split(',')[1]
            img = Image.open(BytesIO(base64.b64decode(uri_data)))

            # NOTE: image might not have a name
            self._name = image_entry['name'] if 'name' in image_entry else 'image_{}.{}'.format(image_index, img.format.lower())
            self._image_path = os.path.join(gltf_loader.root_dir, self._name)
            img.save(self._image_path)
        else:
            self._uri = image_entry['uri']
            self._name = ntpath.basename(self._uri)
            self._image_path = os.path.join(gltf_loader.root_dir, self._uri)

    def get_image_path(self):
        return self._image_path

    def write_to_directory(self, output_dir, channels, texture_prefix):
        file_name = '{0}_{1}'.format(texture_prefix, ntpath.basename(self._name)) if texture_prefix else ntpath.basename(self._name)
        destination = os.path.join(output_dir, file_name)
        original_img = Image.open(self._image_path)
        img = original_img
        if img.mode == 'P':
            img = img.convert('RGBA')
        img_channels = img.split()
        if channels == ImageColorChannels.RGB:
            img = Image.merge('RGB', (img_channels[0], img_channels[1], img_channels[2]))
        elif channels == ImageColorChannels.RGBA:
            img = original_img.convert('RGBA')
        elif channels == ImageColorChannels.R or channels == ImageColorChannels.G or channels == ImageColorChannels.B or channels == ImageColorChannels.A:
            if img.mode != 'L':
                img = img.getchannel(channels.value)

        else:
            raise Exception('Unsupported image channel format {}'.format(channels))

        img.save(destination)
        
        return file_name 


