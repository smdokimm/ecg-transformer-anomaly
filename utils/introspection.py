import torch.nn as nn   
# Stores output shapes for each named module
layer_shapes = {}

def make_hook(name):
    def hook(module, inp, out):
        # record the output tensor shape
        layer_shapes[name] = tuple(out.shape)
    return hook


def register_hooks(model: nn.Module):
    """
    Attach forward-hooks to all submodules (except the top-level)
    to capture their output shapes into layer_shapes.
    """
    for name, module in model.named_modules():
        if name:  # skip the top-level module itself
            module.register_forward_hook(make_hook(name))