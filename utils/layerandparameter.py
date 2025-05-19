import torch
import torch.nn as nn
from collections import defaultdict

default_layer_shapes = {}

def make_hook(name):
    def hook(module, inp, out):
        default_layer_shapes[name] = tuple(out.shape)
    return hook

def summarize_model(model: nn.Module, config: dict, device: torch.device):
    for name, module in model.named_modules():
        if name:
            module.register_forward_hook(make_hook(name))
    seq_len = config['seq_len']
    in_ch = config['in_channels']
    dummy = torch.zeros(1, seq_len, in_ch, device=device)
    model(dummy)
    print("=== Layer Output Shapes ===")
    for name, shape in default_layer_shapes.items():
        print(f"{name:50s} -> {shape}")
    print("\n=== Named Parameters ===")
    module_counts = defaultdict(int)
    total_params = 0
    for name, param in model.named_parameters():
        cnt = param.numel()
        total_params += cnt
        prefix = '.'.join(name.split('.')[:-1])
        module_counts[prefix] += cnt
        shape = tuple(param.shape)
        print(f"{name:50s} | shape: {shape!s:15s} | params: {cnt:,}")

    print("\n=== Params by Module ===")
    for mod, cnt in sorted(module_counts.items()):
        print(f"{mod:30s}: {cnt:,}")

    print(f"\nTotal params:     {total_params:,}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
