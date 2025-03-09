import torch
import torch.nn as nn
import math
from functools import reduce
from operator import mul

class CustomModel(nn.Module):
    def __init__(self, layers_list):
        super(CustomModel, self).__init__()
        self.layers = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.layers(x)

def conv2d_handler(input_shape, params, batch_norm, dropout, activation_type):
    in_channels, in_h, in_w = input_shape
    f_h, f_w = params['f']
    s_h, s_w = params['s']
    p_h, p_w = params['p']
    out_channels = params['out']

    out_h = math.floor((in_h + 2 * p_h - f_h) / s_h) + 1
    out_w = math.floor((in_w + 2 * p_w - f_w) / s_w) + 1

    mul_ops = f_h * f_w * in_channels
    add_ops = mul_ops - 1

    num_mul_ops = mul_ops * out_h * out_w
    num_add_ops = add_ops * out_h * out_w
    num_weights = f_h * f_w * in_channels * out_channels

    model_layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=(f_h, f_w), stride=(s_h, s_w), padding=(p_h, p_w))
    model_layers.append(conv_layer)

    if batch_norm:
        model_layers.append(nn.BatchNorm2d(out_channels))

    if activation_type:
        model_layers.append(activations[activation_type])

    if dropout:
        model_layers.append(nn.Dropout(p=dropout))

    return (out_channels, out_h, out_w), model_layers, num_mul_ops, num_add_ops, num_weights

def pool2d_handler(input_shape, params, batch_norm, dropout, activation_type):
    in_channels, in_h, in_w = input_shape
    k_h, k_w = params.get('f', params.get('kernel_size'))
    s_h, s_w = params.get('s', params.get('stride', (k_h, k_w )))
    p_h, p_w = params.get('p', params.get('padding', [0, 0]))

    out_h = (in_h + 2 * p_h - k_h) // s_h + 1
    out_w = (in_w + 2 * p_w - k_w) // s_w + 1

    num_mul_ops, num_add_ops, num_weights = 0, 0, 0

    model_layers = []
    pool_layer = nn.MaxPool2d(kernel_size=(k_h, k_w), stride=(s_h, s_w), padding=(p_h, p_w))
    model_layers.append(pool_layer)

    if batch_norm:
        print("No batch_norm in pool layer!")

    if activation_type:
        model_layers.append(activations[activation_type])

    if dropout:
        model_layers.append(nn.Dropout(p=dropout))

    return (in_channels, out_h, out_w), model_layers, num_mul_ops, num_add_ops, num_weights

def linear_handler(input_shape, params, batch_norm, dropout, activation_type):
    model_layers = []
    if isinstance(input_shape, tuple) and len(input_shape) > 1:
        flat_dim = reduce(mul, input_shape)
        model_layers.append(nn.Flatten())
    else:
        flat_dim = input_shape

    linear_layer = nn.Linear(flat_dim, params['out'])
    model_layers.append(linear_layer)

    if batch_norm:
        model_layers.append(nn.BatchNorm2d(params['out']))

    if activation_type:
        model_layers.append(activations[activation_type])

    if dropout:
        model_layers.append(nn.Dropout(p=dropout))

    num_mul_ops = flat_dim * params['out']  # Для каждого выходного нейрона нужно сделать умножение на каждый вход
    num_add_ops = num_mul_ops - params['out']  # Сложений будет на 1 меньше, чем умножений
    num_weights = flat_dim * params['out']  # Количество весов

    return (params['out'],), model_layers, num_mul_ops, num_add_ops, num_weights

handlers = {
    'Conv2d': conv2d_handler,
    'Linear': linear_handler,
    'MaxPool2d': pool2d_handler,
    'AvgPool2d': pool2d_handler
}

activations = {
        'ReLU': nn.ReLU(),
        'Sigmoid': nn.Sigmoid(),
        'LeakyReLU': nn.LeakyReLU(0.1),
        'SoftSign': nn.Softsign()
    }

def build_torch_model(arch, input_shape):
    layers = arch.get('layers', {})
    connections = arch.get('connections', {})

    shapes = {"input": input_shape}
    model_layers = []

    current = "input"
    total_mul_ops, total_add_ops, total_weights = 0, 0, 0

    while True:
        prev_layers = []
        for layer_name, prev in connections.items():
            if isinstance(prev, list):
                if current in prev:
                    prev_layers.append(layer_name)
            elif prev == current:
                prev_layers.append(layer_name)
                break

        if not prev_layers:
            break

        for layer_name in prev_layers:
            layer_spec = layers.get(layer_name, {})
            layer_type = layer_spec.get('type')
            params = layer_spec.get('params', {})
            activation_type = layer_spec.get('activation', None)
            batch_norm = layer_spec.get('batch_norm', False)
            dropout = layer_spec.get('dropout', 0.0)

            if layer_type in handlers:
                input_for_layer = shapes[current]
                new_shape, layer, mul_ops, add_ops, num_weights = handlers[layer_type](input_for_layer, params, batch_norm, dropout, activation_type)

                print(f"{layer_name} ({layer_type}): {input_for_layer} -> {new_shape} w:{num_weights}, *:{mul_ops}, +:{add_ops}")

                if isinstance(layer, list):
                    model_layers.extend(layer)
                else:
                    model_layers.append(layer)

                shapes[layer_name] = new_shape

                total_mul_ops += mul_ops
                total_add_ops += add_ops
                total_weights += num_weights
            else:
                print(f"Предупреждение: нет обработчика для слоя типа {layer_type}. Форма не изменена.")
                shapes[layer_name] = shapes[current]

            current = layer_name

    print(f"Общее количество умножений: {total_mul_ops}")
    print(f"Общее количество сложений: {total_add_ops}")
    print(f"Общее количество весов: {total_weights}")

    return CustomModel(model_layers), shapes

def print_network_shapes(shapes):
    print("\nРазмеры слоёв нейросети:")
    for layer, shape in shapes.items():
        if isinstance(shape, tuple) and len(shape) == 3:
            print(f"{layer}: {shape[0]} каналов, высота {shape[1]}, ширина {shape[2]}")
        else:
            print(f"{layer}: {shape[0]} выходных нейронов")

def configure_model(arch, input_shape):
    model, shapes = build_torch_model(arch, input_shape)
    
    # print_network_shapes(shapes)
    # print("\nСгенерированная модель в формате PyTorch:")
    # print(model)
    return model

if __name__ == '__main__':
    arch = {
      "layers": {
        "conv0": {"type": "Conv2d", "params": {"in": 1, "out": 16, "f": [3, 3], "s": [1, 1], "p": [0, 0]}, "activation": "ReLU"},
        "conv1": {"type": "Conv2d", "params": {"in": 16, "out": 16, "f": [5, 5], "s": [2, 2], "p": [2, 2]}, "activation": "ReLU"},
        "conv2": {"type": "Conv2d", "params": {"in": 16, "out": 16, "f": [3, 3], "s": [1, 1], "p": [1, 1]}, "activation": "ReLU"},
        "pool1": {"type": "MaxPool2d", "params": {"f": [2, 2], "s": [2, 2], "p": [0, 0]}},
        "conv3": {"type": "Conv2d", "params": {"in": 16, "out": 24, "f": [5, 5], "s": [2, 2], "p": [2, 2]}, "activation": "ReLU"},
        "conv4": {"type": "Conv2d", "params": {"in": 24, "out": 24, "f": [3, 3], "s": [1, 1], "p": [1, 1]}, "activation": "ReLU"},
        "conv5": {"type": "Conv2d", "params": {"in": 24, "out": 24, "f": [3, 3], "s": [1, 1], "p": [1, 1]}, "activation": "ReLU"},
        "output": {"type": "Linear", "params":{"out": 25}, "activation": "Sigmoid"}
      },
      "connections": {
        "conv0": "input",
        "conv1": "conv0",
        "conv2": "conv1",
        "pool1": "conv2",
        "conv3": "pool1",
        "conv4": "conv3",
        "conv5": "conv4",
        "output": "conv5"
      }
    }
    
    input_shape = (1, 28, 28)
    model = configure_model(arch, input_shape)

