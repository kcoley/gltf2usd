from bisect import bisect_left

from _gltf2usd.gltf2usdUtils import GLTF2USDUtils

from pxr import Gf

class AnimationSampler:
    def __init__(self, sampler_entry, animation):
        self._animation = animation
        self._input_accessor_index = sampler_entry['input']
        self._input_accessor = self._animation._gltf_loader.json_data['accessors'][self._input_accessor_index]
        self._interpolation = sampler_entry['interpolation'] if ('interpolation' in sampler_entry) else 'LINEAR'
        self._output_accessor_index = sampler_entry['output']
        self._output_accessor = self._animation._gltf_loader.json_data['accessors'][self._output_accessor_index]
        self._input_count = self._input_accessor['count']
        self._input_min = self._input_accessor['min']
        self._input_max = self._input_accessor['max']
        self._output_count = self._output_accessor['count']
        self._output_min = self._output_accessor['min'] if ('min' in self._output_accessor) else None
        self._output_max = self._output_accessor['max'] if ('max' in self._output_accessor) else None
        self._input_data = None
        self._output_data = None

        self._input_data = None
        self._output_data = None

    def get_input_count(self):
        return self._input_count

    def get_input_min(self):
        return self._input_min

    def get_input_max(self):
        return self._input_max

    def get_output_count(self):
        return self._output_count

    def get_output_min(self):
        return self._output_min

    def get_output_max(self):
        return self._output_max
        
    def get_input_data(self):
        if not self._input_data:
            accessor = self._animation._gltf_loader.json_data['accessors'][self._input_accessor_index]
            self._input_data = self._animation._gltf_loader.get_data(accessor)
        
        return self._input_data

    def get_output_data(self):
        if not self._output_data:
            accessor = self._animation._gltf_loader.json_data['accessors'][self._output_accessor_index]
            self._output_data = self._animation._gltf_loader.get_data(accessor)
        
        return self._output_data
        

    def get_interpolated_output_data(self, input_sample):
        input_data = self.get_input_data()
        output_data = self.get_output_data()

        closest_pos = bisect_left(input_data, input_sample)
        if closest_pos == 0:
            value = output_data[0]
            if len(value) == 4:
                return Gf.Quatf(value[3], value[0], value[1], value[2])
            else:
                return value
        elif closest_pos == len(input_data):
            value = output_data[-1]
            if len(value) == 4:
                return Gf.Quatf(value[3], value[0], value[1], value[2])
            else:
                return value
        else:
            left_output_sample = output_data[closest_pos - 1]
            right_output_sample = output_data[closest_pos]

            factor = float(input_sample - input_data[closest_pos-1])/(input_data[closest_pos] - input_data[closest_pos - 1])
            if self._interpolation == 'LINEAR':
                return self._linear_interpolate_values(left_output_sample, right_output_sample, factor)
            elif self._interpolation == 'STEP':
                return self._step_interpolate_values(left_output_sample, right_output_sample, factor)
            else:
                print('cubic spline interpolation not yet implemented!  Defaulting to linear for now...')
                return self._linear_interpolate_values(left_output_sample, right_output_sample, factor)

    def _linear_interpolate_values(self, value0, value1, factor):
        if len(value0) == 3:
            one_minus_factor = 1 - factor
            #translation or scale interpolation
            return [
                (factor * value0[0] + (one_minus_factor * value1[0])), 
                (factor * value0[1] + (one_minus_factor * value1[1])), 
                (factor * value0[2] + (one_minus_factor * value1[2]))
            ]

        elif len(value0) == 4:
            #quaternion interpolation
            result = GLTF2USDUtils.slerp(value0, value1, factor)
            return result
        else:
            raise Exception('unsupported value type')

    def _step_interpolate_values(self, value0, value1, factor):
        if len(value0) == 3:
            #translation or scale interpolation
            return value0

        elif len(value0) == 4:
            #quaternion interpolation
            return Gf.Quatf(value0[3], value0[0], value0[1], value0[2])
        else:
            raise Exception('unsupported value type')
            
 

class AnimationChannelTarget:
    def __init__(self, animation_channel_target_entry):
        self._node_index = animation_channel_target_entry['node']
        self._path = animation_channel_target_entry['path']
    @property
    def path(self):
        return self._path

class AnimationChannel:
    def __init__(self, channel_entry, animation):
        self._sampler_index = channel_entry['sampler']
        self._target = AnimationChannelTarget(channel_entry['target'])
        self._animation = animation

    @property
    def target(self):
        return self._target

    def get_sampler_index(self):
        return self._sampler_index

    @property
    def sampler(self):
        return self._animation._samplers[self._sampler_index]


class Animation:
    def __init__(self, animation_entry, index, gltf_loader):
        self._gltf_loader = gltf_loader
        self._name = animation_entry['name'] if ('name' in animation_entry) else 'animation_{}'.format(index)
        self._samplers = [AnimationSampler(sampler, self) for sampler in animation_entry['samplers']]
        self._channels = [AnimationChannel(channel, self) for channel in animation_entry['channels']]
        

    def get_animation_channel_for_node_and_path(self, node, path):
        for channel in self._channels:
            if (channel._target._node_index == node.get_index() and channel._target._path == path):
                return channel

        return None

    def get_animation_channels_for_node(self, node):
        return [channel for channel in self._channels if (channel._target._node_index == node.get_index())]

    def get_channels(self):
        return self._channels

    def get_samplers(self):
        return self._samplers

    def get_sampler_at_index(self, index):
        return self._samplers[index]

