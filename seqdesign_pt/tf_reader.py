import torch
from tensorflow.python import pywrap_tensorflow


class ConversionError(Exception):
    pass


class TFReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.reader = pywrap_tensorflow.NewCheckpointReader(filepath)
        self.unused_keys = []
        self.reset_keys()

    def reset_keys(self):
        self.unused_keys = list(self.reader.get_variable_to_shape_map().keys())
        self.unused_keys = [key for key in self.unused_keys if not key.startswith("Backprop")]

    def get_checkpoint_legacy_version(self):
        if 'Forward/Encoder/DilationBlock1/ConvNet1D/Conv8_3x200/DilatedConvGen8/W' in self.unused_keys:
            return 0
        else:
            return 1

    def load_autoregressive_fr(self, model):
        for m, m_name in zip([model.model['model_f'], model.model['model_r']],
                             ['Forward/', 'Reverse/']):
            self.load_autoregressive(m, m_name)

    def load_autoregressive(self, model, model_name):
        self.load_conv2d(model.start_conv, model_name + 'EncoderPrepareInput/Features1D/')

        for i_block, block in enumerate(model.dilation_blocks):
            block_name = model_name + f'Encoder/DilationBlock{i_block + 1}/ConvNet1D/'
            dilation_schedule = block.dilations

            for i_layer, layer in enumerate(block.dilation_layers):
                layer_name = block_name + f'Conv{i_layer}_3x{dilation_schedule[i_layer]}/'

                self.load_layer_norm(layer.layernorm_1, layer_name + 'ScaleAndShift/')
                self.load_layer_norm(layer.layernorm_2, layer_name + 'ScaleShiftDeltaLayer/ScaleAndShift/')

                self.load_conv2d(layer.mix_conv_1, layer_name + f'Mix1{i_layer}/')
                self.load_conv2d(layer.dilated_conv, layer_name + f'DilatedConvGen{i_layer}/')
                self.load_conv2d(layer.mix_conv_3, layer_name + f'Mix3{i_layer}/')

        self.load_conv2d(model.end_conv, model_name + 'WriteSequence/conv2D/')

        model.step = self.reader.get_tensor('global_step')

    def load_conv2d(self, layer, layer_name):
        self.set_parameter(layer.bias, layer_name + 'b')
        self.set_parameter(layer.weight_g, layer_name + 'g')
        self.set_parameter(layer.weight_v, layer_name + 'W', permute=(3, 2, 0, 1))  # HWIO to OIHW

    def load_layer_norm(self, layer, layer_name):
        self.set_parameter(layer.bias, layer_name + 'b')
        self.set_parameter(layer.weight, layer_name + 'g')

    def set_parameter(self, parameter, name='', permute=()):
        new_data = self.reader.get_tensor(name)
        new_data = torch.as_tensor(new_data, dtype=parameter.dtype, device=parameter.device)
        new_data.requires_grad_(True)

        if permute:
            new_data = new_data.permute(*permute).contiguous()

        if new_data.shape != parameter.shape:
            raise ConversionError('mismatched shapes: {} to {} at {}'.format(
                tuple(new_data.shape), tuple(parameter.shape), name))

        parameter.data = new_data
        self.unused_keys.remove(name)


if __name__ == '__main__':
    import sys
    import autoregressive_model
    model_test = autoregressive_model.AutoregressiveFR()
    reader = TFReader(sys.argv[1])
    reader.load_autoregressive_fr(model_test)
    print([key for key in reader.unused_keys if not key.startswith("Backprop")])
