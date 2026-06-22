#!/usr/bin/env python3
"""
Generate buffer configuration lookup table from layer configurations

This script extracts conv2d layers directly from PyTorch models (or reads
from .csv/.json) and:
1. Extracts unique conv2d layer parameters via forward hooks
2. Calculates optimal buffer sizes and tiling for each layer
3. Generates a C lookup table with all configurations
4. Outputs conv_layer_configs.h for runtime use

Usage:
    # Direct from model (no CSV needed):
    python generate_layer_configs.py --model resnet18 --output conv_layer_configs.h --dram0 64000 --dram1 64000
    python generate_layer_configs.py --model resnet50 --output conv_layer_configs.h --dram0 64000 --dram1 64000
    python generate_layer_configs.py --model resnet18+resnet50 --output conv_layer_configs.h --dram0 64000 --dram1 64000

    # From existing CSV
    python generate_layer_configs.py resnet_conv_list.csv --output conv_layer_configs.h --dram0 64000 --dram1 64000

    # From .pte extraction JSON
    python generate_layer_configs.py layers_config.json --dram0 32768 --dram1 32768

    # Generate all configs in no-DMA mode (changes _dma suffix to _no_dma for every kernel name)
    python generate_layer_configs.py resnet_conv_list.csv --output conv_layer_configs_no_dma.h --dram0 64000 --dram1 64000 --no-dma-mode
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import OrderedDict

# Import the existing buffer calculation logic
sys.path.insert(0, str(Path(__file__).parent))
from generate_idma_buffers import (
    find_max_tile_config,
    calculate_buffer_sizes_with_rows,
    calculate_buffer_placement,
    DRAM_SIZE_0,
    DRAM_SIZE_1
)

# ---------------------------------------------------------------------------
# Direct model extraction (replaces extract_resnet_layers.py)
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = ['resnet18', 'resnet50']


def _build_name_map_resnet18():
    """ResNet-18 (BasicBlock): 2 conv layers per block, 2 blocks per layer group."""
    m = OrderedDict()
    m['conv1'] = 'conv1'
    m['layer1.0.conv1'] = 'conv2.1'
    m['layer1.0.conv2'] = 'conv2.2'
    m['layer1.1.conv1'] = 'conv3.1'
    m['layer1.1.conv2'] = 'conv3.2'
    m['layer2.0.downsample.0'] = 'conv4a.1'
    m['layer2.0.conv1'] = 'conv4b.1'
    m['layer2.0.conv2'] = 'conv4b.2'
    m['layer2.1.conv1'] = 'conv5.1'
    m['layer2.1.conv2'] = 'conv5.2'
    m['layer3.0.downsample.0'] = 'conv6a.1'
    m['layer3.0.conv1'] = 'conv6b.1'
    m['layer3.0.conv2'] = 'conv6b.2'
    m['layer3.1.conv1'] = 'conv7.1'
    m['layer3.1.conv2'] = 'conv7.2'
    m['layer4.0.downsample.0'] = 'conv8a.1'
    m['layer4.0.conv1'] = 'conv8b.1'
    m['layer4.0.conv2'] = 'conv8b.2'
    m['layer4.1.conv1'] = 'conv9.1'
    m['layer4.1.conv2'] = 'conv9.2'
    return m


def _build_name_map_resnet50():
    """ResNet-50 (Bottleneck): 3 conv layers per block, variable blocks per layer group."""
    m = OrderedDict()
    m['conv1'] = 'conv1'
    layer_blocks = {1: 3, 2: 4, 3: 6, 4: 3}
    conv_counter = 2
    for layer_idx in range(1, 5):
        n_blocks = layer_blocks[layer_idx]
        for blk in range(n_blocks):
            prefix = f'layer{layer_idx}.{blk}'
            has_ds = (blk == 0)
            if has_ds:
                m[f'{prefix}.downsample.0'] = f'conv{conv_counter}a.1'
                m[f'{prefix}.conv1'] = f'conv{conv_counter}b.1'
                m[f'{prefix}.conv2'] = f'conv{conv_counter}b.2'
                m[f'{prefix}.conv3'] = f'conv{conv_counter}b.3'
            else:
                m[f'{prefix}.conv1'] = f'conv{conv_counter}.1'
                m[f'{prefix}.conv2'] = f'conv{conv_counter}.2'
                m[f'{prefix}.conv3'] = f'conv{conv_counter}.3'
            conv_counter += 1
    return m


def _get_conv_layers_via_hooks(model, name_map, input_size=(1, 3, 64, 64)):
    """
    Run forward hooks on every Conv2d layer to capture input/output shapes
    and convolution parameters.  Returns OrderedDict keyed by friendly name.
    """
    import torch
    import torch.nn as nn

    layers_info = OrderedDict()
    hooks = []

    def make_hook(friendly_name):
        def hook_fn(module, inp, out):
            layers_info[friendly_name] = {
                'input': list(inp[0].shape),
                'kernel': list(module.weight.shape),
                'stride': list(module.stride),
                'padding': list(module.padding),
                'dilation': list(module.dilation),
                'transposed': isinstance(module, nn.ConvTranspose2d),
                'output_padding': (list(module.output_padding)
                                   if hasattr(module, 'output_padding') else [0, 0]),
                'groups': module.groups,
                'output': list(out.shape),
            }
        return hook_fn

    for mod_name, module in model.named_modules():
        if mod_name in name_map and isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(module.register_forward_hook(make_hook(name_map[mod_name])))

    x = torch.randn(*input_size)
    with torch.no_grad():
        model.eval()
        model(x)

    for h in hooks:
        h.remove()
    return layers_info


def _make_unique_key(info):
    """Hashable key for deduplication across models."""
    return (
        tuple(info['input']),
        tuple(info['kernel']),
        tuple(info['stride']),
        tuple(info['padding']),
        tuple(info['dilation']),
        info['transposed'],
        tuple(info['output_padding']),
        info['groups'],
        tuple(info['output']),
    )


def load_layers_from_model(model_names, input_size=(1, 3, 64, 64)):
    """
    Extract unique conv2d layers directly from one or more torchvision models.

    Args:
        model_names: list of model name strings, e.g. ['resnet18', 'resnet50']
        input_size:  tuple for the dummy forward pass, e.g. (1, 3, 64, 64)

    Returns:
        list of layer dicts in the internal format expected by calculate_layer_config()
    """
    import torch  # noqa: deferred import so torch is only needed when --model is used

    builders = {
        'resnet18': ('torchvision.models', 'resnet18', 'ResNet18_Weights', _build_name_map_resnet18),
        'resnet50': ('torchvision.models', 'resnet50', 'ResNet50_Weights', _build_name_map_resnet50),
    }

    seen_keys = set()
    unique_layers = []  # (friendly_name, info_dict, source_model)

    for model_name in model_names:
        model_name = model_name.strip().lower()
        if model_name not in builders:
            raise ValueError(f"Unsupported model '{model_name}'. Supported: {list(builders.keys())}")

        mod_path, fn_name, wt_name, name_map_fn = builders[model_name]
        print(f"Loading {model_name}...")
        import importlib
        tv = importlib.import_module(mod_path)
        build_fn = getattr(tv, fn_name)
        weights = getattr(tv, wt_name).DEFAULT
        model = build_fn(weights=weights)
        model.eval()

        name_map = name_map_fn()
        layers_info = _get_conv_layers_via_hooks(model, name_map, input_size)

        for name, info in layers_info.items():
            key = _make_unique_key(info)
            if key not in seen_keys:
                seen_keys.add(key)
                unique_layers.append((name, info, model_name))

    print(f"Extracted {len(unique_layers)} unique conv layers from {', '.join(model_names)}")

    # Convert to the internal layer-dict format used by calculate_layer_config()
    layers = []
    for layer_id, (name, info, _source) in enumerate(unique_layers):
        _, in_c, in_h, in_w = info['input']
        _, out_c, out_h, out_w = info['output']
        _oc, in_channels, k_h, k_w = info['kernel']
        layers.append({
            'layer_id': layer_id,
            'name': name,
            'input': (in_w, in_h, in_c),
            'output': (out_w, out_h, out_c),
            'kernel': (k_w, k_h, in_channels, _oc),
            'stride': tuple(info['stride']),
            'padding': tuple(info['padding']),
            'dilation': tuple(info['dilation']),
        })
    return layers


# ---------------------------------------------------------------------------
# PTE-based loader (ExecuTorch .pte binary via exir source tree)
# ---------------------------------------------------------------------------

# Default paths relative to this script's location
# backends/cadence/vision/config_generator/ → .parent×5 → ext_test/executorch
_EXECUTORCH_SRC = str(Path(__file__).parent.parent.parent.parent.parent)  # ext_test/executorch
_EXECUTORCH_PARENT = str(Path(__file__).parent.parent.parent.parent.parent.parent)  # ext_test
_FLATC_DEFAULT = str(Path(__file__).parent.parent.parent.parent.parent /
                     "cmake-out/third-party/flatc_ep/bin/flatc")


def _bootstrap_executorch_imports(flatc_path=None):
    """
    Bootstrap executorch.exir from the local source tree without a pip install.

    Bypasses exir/__init__.py (which pulls in many optional deps) by pre-populating
    sys.modules with lightweight stub packages for 'executorch' and 'executorch.exir'.
    Only the _serialize sub-package is actually loaded.

    Also sets FLATC_EXECUTABLE so _flatbuffer.py can find the flatc binary.
    """
    import types

    # Add ext_test/ so `import executorch…` works, and ext_test/executorch/ so
    # internal sub-imports like `from executorch.exir._serialize…` resolve correctly.
    if _EXECUTORCH_PARENT not in sys.path:
        sys.path.insert(0, _EXECUTORCH_PARENT)
    if _EXECUTORCH_SRC not in sys.path:
        sys.path.insert(0, _EXECUTORCH_SRC)

    # Stub 'executorch' and 'executorch.exir' so Python never runs their
    # __init__.py files (which have heavy, optional dependencies).
    for pkg, pkg_dir in [
        ('executorch',       _EXECUTORCH_SRC),
        ('executorch.exir',  _EXECUTORCH_SRC + '/exir'),
    ]:
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [pkg_dir]
            m.__package__ = pkg
            sys.modules[pkg] = m

    # Tell _flatbuffer.py where to find the flatc binary.
    resolved = flatc_path or _FLATC_DEFAULT
    if os.path.isfile(resolved):
        os.environ.setdefault('FLATC_EXECUTABLE', resolved)


def load_layers_from_pte(pte_file, flatc_path=None):
    """
    Extract unique conv2d layers directly from an ExecuTorch .pte binary.

    Mirrors load_layers_from_model() but reads the serialised execution plan
    instead of running a live forward pass.  Works without a full executorch
    pip install by loading the _serialize sub-package from the local source
    tree.

    Args:
        pte_file:   Path to the .pte file (str or Path).
        flatc_path: Optional path to the flatc binary.  Defaults to the
                    cmake-out copy built alongside the source tree.

    Returns:
        list of layer dicts in the internal format expected by
        calculate_layer_config(), same as load_layers_from_model().
    """
    _bootstrap_executorch_imports(flatc_path)

    from executorch.exir._serialize._program import deserialize_pte_binary
    from executorch.exir.schema import KernelCall, Int, IntList, Tensor

    pte_path = Path(pte_file)
    print(f"Loading PTE: {pte_path} ...")

    with open(pte_path, 'rb') as f:
        pte_file_obj = deserialize_pte_binary(f.read())

    # deserialize_pte_binary returns a PTEFile wrapper; unwrap to get Program
    if hasattr(pte_file_obj, 'program'):
        program = pte_file_obj.program
    else:
        program = pte_file_obj  # older API returned Program directly

    plan   = program.execution_plan[0]
    values = plan.values

    # ------------------------------------------------------------------
    # Helpers to dereference EValue indices from the values table
    # ------------------------------------------------------------------
    def _tensor(idx):
        v = values[idx].val
        return v if isinstance(v, Tensor) else None

    def _int_val(idx):
        v = values[idx].val
        return v.int_val if isinstance(v, Int) else None

    def _intlist_val(idx):
        """IntList.items are EValue indices pointing to Int EVals."""
        v = values[idx].val
        if isinstance(v, IntList):
            return [_int_val(i) for i in v.items]
        return None

    # ------------------------------------------------------------------
    # Walk all KernelCall instructions and collect conv layers
    # ------------------------------------------------------------------
    # cadence::quantized_conv2d_nchw arg order (from quantized_conv2d_nchw_out.cpp):
    #   [0] input  [1] weight  [2] bias
    #   [3] stride  [4] padding  [5] dilation
    #   [6] groups  [7] in_zero_point  … [−2/−1] out
    CONV_OPS = {
        'cadence::quantized_conv2d_nchw',
        'aten::conv2d',
        'aten::convolution',
    }

    seen_keys = set()
    unique_layers = []

    for instr in plan.chains[0].instructions:
        ia = instr.instr_args
        if not isinstance(ia, KernelCall):
            continue
        op_name = plan.operators[ia.op_index].name
        if op_name not in CONV_OPS:
            continue

        args = ia.args
        input_t  = _tensor(args[0])
        weight_t = _tensor(args[1])
        output_t = _tensor(args[-1])   # last arg is always the output tensor

        if input_t is None or weight_t is None or output_t is None:
            continue

        stride   = _intlist_val(args[3]) or [1, 1]
        padding  = _intlist_val(args[4]) or [0, 0]
        dilation = _intlist_val(args[5]) or [1, 1]

        # shapes are NCHW
        _, in_c,  in_h,  in_w  = input_t.sizes
        _, out_c, out_h, out_w = output_t.sizes
        _oc, _ic, k_h, k_w    = weight_t.sizes

        info = {
            'input':   (in_w,  in_h,  in_c),
            'output':  (out_w, out_h, out_c),
            'kernel':  (k_w,   k_h,   _ic,   _oc),
            'stride':  tuple(stride),
            'padding': tuple(padding),
            'dilation':tuple(dilation),
        }

        key = (info['input'], info['output'], info['kernel'],
               info['stride'], info['padding'], info['dilation'])
        if key not in seen_keys:
            seen_keys.add(key)
            unique_layers.append(info)

    print(f"Extracted {len(unique_layers)} unique conv layers from PTE")

    # Convert to the internal layer-dict format (same as load_layers_from_model)
    layers = []
    for layer_id, info in enumerate(unique_layers):
        in_w,  in_h,  in_c  = info['input']
        out_w, out_h, out_c = info['output']
        k_w,   k_h,   _ic,  _oc = info['kernel']
        # Derive a friendly name from the kernel shape
        name = f"conv_{k_h}x{k_w}_s{info['stride'][0]}_ic{in_c}_oc{out_c}"
        layers.append({
            'layer_id':  layer_id,
            'name':      name,
            'input':     info['input'],
            'output':    info['output'],
            'kernel':    info['kernel'],
            'stride':    info['stride'],
            'padding':   info['padding'],
            'dilation':  info['dilation'],
        })
    return layers


# ---------------------------------------------------------------------------
# File-based loaders (CSV / JSON)
# ---------------------------------------------------------------------------

def load_layers_from_json(json_file):
    """Load layer configurations from JSON file"""
    with open(json_file, 'r') as f:
        return json.load(f)

def load_layers_from_csv(csv_file):
    """Load layer configurations from ResNet CSV file (tab-delimited)"""
    import csv
    
    layers = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        layer_id = 0
        
        for row in reader:
            # Skip header or empty rows
            if not row or not row[0].strip() or 'input' in row[0].lower() or (len(row) > 1 and 'input' in row[1].lower()):
                continue
            
            # Tab-delimited format: layer_name \t input \t kernel \t stride \t padding \t dilation \t transposed \t output_padding \t groups \t output
            layer_name = row[0].strip()
            
            # Parse shapes from CSV
            input_shape = tuple(int(x) for x in row[1].strip().split(','))   # e.g., "1,3,64,64"
            kernel_shape = tuple(int(x) for x in row[2].strip().split(','))  # e.g., "64,3,7,7"
            stride = tuple(int(x) for x in row[3].strip().split(','))        # e.g., "2, 2"
            padding = tuple(int(x) for x in row[4].strip().split(',')) if len(row) > 4 else (0, 0)
            output_shape = tuple(int(x) for x in row[9].strip().split(','))  # e.g., "1,64,32,32"
            
            # Convert to internal format
            _, in_c, in_h, in_w = input_shape
            _, out_c, out_h, out_w = output_shape
            out_channels, in_channels, k_h, k_w = kernel_shape
            
            layer = {
                'layer_id': layer_id,
                'name': layer_name,
                'input': (in_w, in_h, in_c),
                'output': (out_w, out_h, out_c),
                'kernel': (k_w, k_h, in_channels, out_channels),
                'stride': tuple(stride),
                'padding': tuple(padding),
                'dilation': (1, 1)
            }
            
            layers.append(layer)
            layer_id += 1
    
    return layers

def calculate_layer_config(layer, dram0_size, dram1_size):
    """
    Calculate complete buffer configuration for a single layer
    
    Returns: Dictionary with all runtime parameters
    """
    # Unpack layer parameters
    input_w, input_h, input_c = layer['input']
    output_w, output_h, output_c = layer['output']
    kernel_w, kernel_h, in_c, out_c = layer['kernel']
    stride_w, stride_h = layer['stride']
    pad_w, pad_h = layer['padding']
    
    # Calculate padding edges
    padding = (pad_w, pad_w, pad_h, pad_h, 0, 0)  # (dim1_e1, dim1_e2, dim2_e1, dim2_e2, ...)
    
    # Dummy conv_params (will be set per-model)
    conv_params = (stride_w, stride_h, 8, 4000, 11, 0, 1, kernel_h, kernel_w)
    
    # Generate kernel name based on size and stride
    if kernel_h == 7 and kernel_w == 7 and stride_h == 2:
        kernel_name = "7x7j2d1"
    elif kernel_h == 3 and kernel_w == 3 and stride_h == 1:
        kernel_name = "3x3j1d1"
    elif kernel_h == 3 and kernel_w == 3 and stride_h == 2:
        kernel_name = "3x3j2d1"
    elif kernel_h == 1 and kernel_w == 1 and stride_h == 2:
        kernel_name = "1x1j2d1"
    elif kernel_h == 1 and kernel_w == 1 and stride_h == 1:
        kernel_name = "1x1j1d1"
    else:
        kernel_name = f"{kernel_w}x{kernel_h}j{stride_w}d1"
    
    # Find optimal tiling configuration
    n_tile_size, output_rows, buffer_sizes = find_max_tile_config(
        input_whd=(input_w, input_h, input_c),
        output_whd=(output_w, output_h, output_c),
        kernel_whdn=(kernel_w, kernel_h, in_c, out_c),
        padding=padding,
        stride_xy=(stride_w, stride_h),
        kernel_name=kernel_name,
        data_type="S8S8",
        dram0_size=dram0_size,
        dram1_size=dram1_size,
        conv_params=conv_params
    )
    
    if buffer_sizes is None or n_tile_size == 0 or output_rows == 0:
        print(f"WARNING: Could not find valid DMA configuration for layer {layer['layer_id']} - using cache mode (single tile)")
        
        # Calculate pitches with padding for cache mode
        # in_dim1_size = src_dim1_size (actual input width)
        # in_dim1_pitch = input_w + 2*pad_w (width including padding)
        in_dim1_pitch = input_w + 2 * pad_w
        in_dim2_pitch = in_dim1_pitch * (input_h + 2 * pad_h)
        out_dim1_pitch = output_w
        out_dim2_pitch = out_dim1_pitch * output_h
        coeff_dim1_pitch = kernel_w
        coeff_dim2_pitch = coeff_dim1_pitch * kernel_h
        coeff_dim3_pitch = coeff_dim2_pitch * in_c
        
        # Calculate buffer sizes for full tile (no tiling - process entire layer)
        input_buffer_size = in_dim2_pitch * input_c
        output_buffer_size = out_dim2_pitch * output_c
        coeff_buffer_size = coeff_dim3_pitch * output_c
        bias_buffer_size = output_c * 4  # S32
        outscale_buffer_size = output_c * 2  # U16
        
        # Data offset is 0 for cache mode (no pre-allocated padding in buffer)
        in_data_offset = 0
        
        # Return cache-mode config: single tile processing entire layer
        return {
            'layer_id': layer['layer_id'],
            'layer_name': layer['name'],
            'kernel_name': kernel_name + "_no_dma",
            'src_dim1_size': input_w, 'src_dim2_size': input_h, 'src_dim3_size': input_c,
            'src_dim1_pitch': input_w, 'src_dim2_pitch': input_w * input_h,
            'dst_dim1_size': output_w, 'dst_dim2_size': output_h, 'dst_dim3_size': output_c,
            'dst_dim1_pitch': output_w, 'dst_dim2_pitch': output_w * output_h,
            'in_dim1_size': input_w, 'in_dim1_pitch': in_dim1_pitch,
            'in_dim2_size': input_h, 'in_dim2_pitch': in_dim2_pitch,
            'in_dim1_edge1': pad_w, 'in_dim1_edge2': pad_w, 'in_dim2_edge1': pad_h, 'in_dim2_edge2': pad_h,
            'in_dim3_edge1': 0, 'in_dim3_edge2': 0, 'in_data_offset': in_data_offset, 'in_rows_firstdma': input_h,
            'out_dim1_size': output_w, 'out_dim1_pitch': out_dim1_pitch,
            'out_dim2_size': output_h, 'out_dim2_pitch': out_dim2_pitch, 'out_dim3_size': output_c,
            'coeff_dim1_size': kernel_w, 'coeff_dim2_size': kernel_h, 'coeff_dim3_size': in_c, 'coeff_dim4_size': output_c,
            'coeff_dim1_pitch': coeff_dim1_pitch, 'coeff_dim2_pitch': coeff_dim2_pitch, 'coeff_dim3_pitch': coeff_dim3_pitch,
            'bias_dim1_size': output_c, 'bias_dim2_size': 1,
            'outscale_dim1_size': output_c, 'outscale_dim2_size': 1,
            'input_buffer_size': input_buffer_size, 'coeff_buffer_size': coeff_buffer_size,
            'output_buffer_size': output_buffer_size,
            'bias_buffer_size': bias_buffer_size, 'outscale_buffer_size': outscale_buffer_size,
            'input_ping_dram': 0, 'input_pong_dram': 0, 'coeff_dram': 0,
            'output_ping_dram': 0, 'output_pong_dram': 0, 'bias_dram': 0, 'outscale_dram': 0,
            'n_tile_size': output_c, 'n_tiles': 1, 'n_tile_size_last': output_c, 'height_tiles': 1,
            'output_rows': output_h, 'input_rows': input_h,
            'stride_x': stride_w, 'stride_y': stride_h, 'accum_shift': 8, 'relu_max': 4000,
            'relu_min': 0, 'output_shift': 11, 'output_scale': 0, 'dilation': 1,
            'kernel_w': kernel_w, 'kernel_h': kernel_h, 'padding': pad_w, 'flags': 0,
            'input_zero_point': 0,
            # Generate unique config key: ic_ih_iw_oc_kh_kw_oh_ow_sy_sx_pad_dil
            'config_key': f"{input_c}_{input_h}_{input_w}_{output_c}_{kernel_h}_{kernel_w}_{output_h}_{output_w}_{stride_h}_{stride_w}_{pad_w}_1",
        }
    
    # Calculate additional derived parameters
    n_tiles = (out_c + n_tile_size - 1) // n_tile_size
    height_tiles = (output_h + output_rows - 1) // output_rows
    input_rows = kernel_h + (output_rows - 1) * stride_h
    
    # Get buffer placement
    placement = calculate_buffer_placement(buffer_sizes, dram0_size, dram1_size)
    
    # Build complete config with all fields from convIdma_buffers.h schema
    config = {
        'layer_id': layer['layer_id'],
        'layer_name': layer['name'],
        'kernel_name': kernel_name + "_dma",
        
        # Source dimensions
        'src_dim1_size': buffer_sizes['SRC_DIM1_SIZE'],
        'src_dim2_size': buffer_sizes['SRC_DIM2_SIZE'],
        'src_dim3_size': buffer_sizes['SRC_DIM3_SIZE'],
        'src_dim1_pitch': buffer_sizes['SRC_DIM1_PITCH'],
        'src_dim2_pitch': buffer_sizes['SRC_DIM2_PITCH'],
        
        # Destination dimensions
        'dst_dim1_size': buffer_sizes['DST_DIM1_SIZE'],
        'dst_dim2_size': buffer_sizes['DST_DIM2_SIZE'],
        'dst_dim1_pitch': buffer_sizes['DST_DIM1_PITCH'],
        'dst_dim2_pitch': buffer_sizes['DST_DIM2_PITCH'],
        'dst_dim3_size': output_c,
        
        # Input tile dimensions
        'in_dim1_size': buffer_sizes['IN_DIM1_SIZE'],
        'in_dim1_pitch': buffer_sizes['IN_DIM1_PITCH'],
        'in_dim2_size': buffer_sizes['IN_DIM2_SIZE'],
        'in_dim2_pitch': buffer_sizes['IN_DIM2_PITCH'],
        'in_dim1_edge1': padding[0],
        'in_dim1_edge2': padding[1],
        'in_dim2_edge1': padding[2],
        'in_dim2_edge2': padding[3],
        'in_dim3_edge1': padding[4],
        'in_dim3_edge2': padding[5],
        'in_data_offset': buffer_sizes['IN_DATA_OFFSET'],
        'in_rows_firstdma': buffer_sizes['IN_ROWS_FIRSTDMA'],
        
        # Output tile dimensions
        'out_dim1_size': buffer_sizes['OUT_DIM1_SIZE'],
        'out_dim1_pitch': buffer_sizes['OUT_DIM1_PITCH'],
        'out_dim2_size': buffer_sizes['OUT_DIM2_SIZE'],
        'out_dim2_pitch': buffer_sizes['OUT_DIM2_PITCH'],
        'out_dim3_size': buffer_sizes['OUT_DIM3_SIZE'],
        
        # Coefficient tile dimensions
        'coeff_dim1_size': buffer_sizes['COEFF_DIM1_SIZE'],
        'coeff_dim2_size': buffer_sizes['COEFF_DIM2_SIZE'],
        'coeff_dim3_size': buffer_sizes['COEFF_DIM3_SIZE'],
        'coeff_dim4_size': buffer_sizes['COEFF_DIM4_SIZE'],
        'coeff_dim1_pitch': buffer_sizes['COEFF_DIM1_PITCH'],
        'coeff_dim2_pitch': buffer_sizes['COEFF_DIM2_PITCH'],
        'coeff_dim3_pitch': buffer_sizes['COEFF_DIM3_PITCH'],
        
        # Bias dimensions
        'bias_dim1_size': buffer_sizes['BIAS_DIM1_SIZE'],
        'bias_dim2_size': buffer_sizes['BIAS_DIM2_SIZE'],
        
        # Output scale dimensions
        'outscale_dim1_size': buffer_sizes['OUTSCALE_DIM1_SIZE'],
        'outscale_dim2_size': buffer_sizes['OUTSCALE_DIM2_SIZE'],
        
        # Buffer sizes
        'input_buffer_size': buffer_sizes['IN'],
        'coeff_buffer_size': buffer_sizes['COEFF'],
        'output_buffer_size': buffer_sizes['OUT'],
        'bias_buffer_size': buffer_sizes['BIAS'],
        'outscale_buffer_size': buffer_sizes['OUTSCALE'],
        
        # Buffer DRAM placement (0 or 1)
        'input_ping_dram': placement.get('IN1_dram', 0),
        'input_pong_dram': placement.get('IN2_dram', 1),
        'coeff_dram': placement.get('COEFF_dram', 0),
        'output_ping_dram': placement.get('OUT1_dram', 1),
        'output_pong_dram': placement.get('OUT2_dram', 1),
        'bias_dram': placement.get('BIAS_dram', 1),
        'outscale_dram': placement.get('OUTSCALE_dram', 1),
        
        # Tiling parameters
        'n_tile_size': buffer_sizes['N_TILE_SIZE'],
        'n_tiles': buffer_sizes['N_TILES'],
        'n_tile_size_last': buffer_sizes['N_TILE_SIZE_LAST'],
        'height_tiles': buffer_sizes['HIGHT_TILES'],
        'output_rows': output_rows,
        'input_rows': input_rows,
        
        # Convolution parameters
        'stride_x': buffer_sizes.get('STRIDEX', stride_w),
        'stride_y': buffer_sizes.get('STRIDEY', stride_h),
        'accum_shift': buffer_sizes.get('ACCUM_SHIFT', 8),
        'relu_max': buffer_sizes.get('RELU_MAX', 4000),
        'relu_min': buffer_sizes.get('RELU_MIN', 0),
        'output_shift': buffer_sizes.get('OUTPUT_SHIFT', 11),
        'output_scale': buffer_sizes.get('OUTPUT_SCALE', 0),
        'dilation': buffer_sizes.get('DILATION', 1),
        'kernel_w': kernel_w,
        'kernel_h': kernel_h,
        'padding': pad_w,  # Symmetric padding
        'flags': buffer_sizes.get('FLAGS', 0),
        'input_zero_point': 0,
    }
    
    # Generate unique config key based on layer parameters
    # Format: ic_ih_iw_oc_kh_kw_oh_ow_sy_sx_pad_dil
    dilation = buffer_sizes.get('DILATION', 1)
    config['config_key'] = f"{in_c}_{input_h}_{input_w}_{out_c}_{kernel_h}_{kernel_w}_{output_h}_{output_w}_{stride_h}_{stride_w}_{pad_w}_{dilation}"
    
    return config

def generate_c_header(configs, output_file, dram0_size=32768, dram1_size=32768, no_dma_mode=False):
    """
    Generate C header file with lookup table
    
    Output: conv_layer_configs.h with:
    - typedef struct conv_layer_config_t
    - const conv_layer_config_t CONV_LAYER_CONFIGS[] = {...};
    - int get_num_conv_layers();
    - const conv_layer_config_t* get_layer_config(int layer_id);
    """
    
    with open(output_file, 'w') as f:
        f.write("""/*
 * conv_layer_configs.h
 *
 * Auto-generated convolution layer configurations
 * Generated from model layer extraction
 *
 * DO NOT EDIT MANUALLY - Regenerate with generate_layer_configs.py
 */

#ifndef CONV_LAYER_CONFIGS_H
#define CONV_LAYER_CONFIGS_H

#include <stdint.h>
#include <stddef.h>  // for NULL

/**
 * Runtime configuration for a single convolution layer
 * Contains all parameters needed to execute the layer
 * Matches convIdma_buffers.h schema
 */
typedef struct {
    // Layer identification
    int layer_id;
    const char* layer_name;
    const char* kernel_name;
    const char* config_key;     // Unique key: ic_ih_iw_oc_kh_kw_oh_ow_sy_sx_pad_dil
    
    // Source (DRAM) dimensions
    int src_dim1_size;      // Input width in DRAM
    int src_dim2_size;      // Input height in DRAM
    int src_dim3_size;      // Input channels in DRAM
    int src_dim1_pitch;     // DRAM row pitch
    int src_dim2_pitch;     // DRAM plane pitch
    
    // Destination (DRAM) dimensions
    int dst_dim1_size;      // Output width in DRAM
    int dst_dim2_size;      // Output height in DRAM
    int dst_dim3_size;      // Output channels in DRAM
    int dst_dim1_pitch;     // DRAM row pitch
    int dst_dim2_pitch;     // DRAM plane pitch
    
    // Input tile (local memory) dimensions
    int in_dim1_size;       // Tile width (with padding)
    int in_dim1_pitch;      // Tile row pitch
    int in_dim2_size;       // Tile height (rows per iteration)
    int in_dim2_pitch;      // Tile plane pitch
    int in_dim1_edge1;      // Left padding
    int in_dim1_edge2;      // Right padding
    int in_dim2_edge1;      // Top padding
    int in_dim2_edge2;      // Bottom padding
    int in_dim3_edge1;      // Channel padding (usually 0)
    int in_dim3_edge2;      // Channel padding (usually 0)
    int in_data_offset;     // Offset to actual data in buffer
    int in_rows_firstdma;   // Rows to transfer in first DMA
    
    // Output tile (local memory) dimensions
    int out_dim1_size;      // Output width
    int out_dim1_pitch;     // Output row pitch
    int out_dim2_size;      // Output rows per iteration
    int out_dim2_pitch;     // Output plane pitch
    int out_dim3_size;      // Output channels per N-tile
    
    // Coefficient tile dimensions
    int coeff_dim1_size;    // Kernel width
    int coeff_dim2_size;    // Kernel height
    int coeff_dim3_size;    // Input channels
    int coeff_dim4_size;    // Output channels (total)
    int coeff_dim1_pitch;   // Kernel row pitch
    int coeff_dim2_pitch;   // Kernel plane pitch (W*H)
    int coeff_dim3_pitch;   // Kernel 3D pitch (W*H*D)
    
    // Bias array dimensions
    int bias_dim1_size;     // Number of bias values
    int bias_dim2_size;     // Always 1
    
    // Output scale array dimensions
    int outscale_dim1_size; // Number of scale values
    int outscale_dim2_size; // Always 1
    
    // Buffer sizes (bytes)
    int input_buffer_size;
    int coeff_buffer_size;
    int output_buffer_size;
    int bias_buffer_size;
    int outscale_buffer_size;
    
    // Buffer DRAM placement (0 = DRAM0, 1 = DRAM1)
    int input_ping_dram;
    int input_pong_dram;
    int coeff_dram;
    int output_ping_dram;
    int output_pong_dram;
    int bias_dram;
    int outscale_dram;
    
    // Tiling parameters
    int n_tile_size;        // Output channels per N-tile
    int n_tiles;            // Total number of N-tiles
    int n_tile_size_last;   // Channels in last N-tile
    int height_tiles;       // Total number of H-tiles
    int output_rows;        // Output rows per H-tile
    int input_rows;         // Input rows needed per H-tile
    
    // Convolution parameters
    int kernel_w;
    int kernel_h;
    int stride_x;
    int stride_y;
    int padding;            // Symmetric padding
    int dilation;
    int accum_shift;        // Accumulator shift
    int relu_max;           // ReLU clamp maximum
    int relu_min;           // ReLU clamp minimum
    int output_shift;       // Output quantization shift
    int output_scale;       // Output scale factor
    int flags;              // Convolution flags
    int input_zero_point;   // Input zero point for padding fill
    
} conv_layer_config_t;

""")
        
        # Generate lookup table
        f.write(f"// Total number of convolution layers\n")
        f.write(f"#define NUM_CONV_LAYERS {len(configs)}\n\n")
        
        # Generate IDMA buffer size macros
        _dram0_macro = 0 if no_dma_mode else dram0_size
        _dram1_macro = 0 if no_dma_mode else dram1_size
        f.write(f" #define IDMA_BUFFER_SIZE_DRAM0 ({_dram0_macro}) // {_dram0_macro // 1024} KB for DRAM0\n")
        f.write(f" #define IDMA_BUFFER_SIZE_DRAM1 ({_dram1_macro}) // {_dram1_macro // 1024} KB for DRAM1\n\n")
        
        f.write("// Layer configuration lookup table\n")
        f.write("static const conv_layer_config_t CONV_LAYER_CONFIGS[] = {\n")
        
        for config in configs:
            f.write("    {\n")
            f.write(f"        .layer_id = {config['layer_id']},\n")
            f.write(f"        .layer_name = \"{config['layer_name']}\",\n")
            f.write(f"        .kernel_name = \"{config['kernel_name']}\",\n")
            f.write(f"        .config_key = \"{config['config_key']}\",\n")
            f.write(f"        \n")
            
            # Source dimensions
            f.write(f"        // Source (DRAM): {config['src_dim1_size']}×{config['src_dim2_size']}×{config['src_dim3_size']}\n")
            f.write(f"        .src_dim1_size = {config['src_dim1_size']},\n")
            f.write(f"        .src_dim2_size = {config['src_dim2_size']},\n")
            f.write(f"        .src_dim3_size = {config['src_dim3_size']},\n")
            f.write(f"        .src_dim1_pitch = {config['src_dim1_pitch']},\n")
            f.write(f"        .src_dim2_pitch = {config['src_dim2_pitch']},\n")
            f.write(f"        \n")
            
            # Destination dimensions
            f.write(f"        // Destination (DRAM): {config['dst_dim1_size']}×{config['dst_dim2_size']}×{config['dst_dim3_size']}\n")
            f.write(f"        .dst_dim1_size = {config['dst_dim1_size']},\n")
            f.write(f"        .dst_dim2_size = {config['dst_dim2_size']},\n")
            f.write(f"        .dst_dim3_size = {config['dst_dim3_size']},\n")
            f.write(f"        .dst_dim1_pitch = {config['dst_dim1_pitch']},\n")
            f.write(f"        .dst_dim2_pitch = {config['dst_dim2_pitch']},\n")
            f.write(f"        \n")
            
            # Input tile dimensions
            f.write(f"        // Input tile: {config['in_dim1_size']}×{config['in_dim2_size']} (edges: {config['in_dim1_edge1']},{config['in_dim1_edge2']},{config['in_dim2_edge1']},{config['in_dim2_edge2']})\n")
            f.write(f"        .in_dim1_size = {config['in_dim1_size']},\n")
            f.write(f"        .in_dim1_pitch = {config['in_dim1_pitch']},\n")
            f.write(f"        .in_dim2_size = {config['in_dim2_size']},\n")
            f.write(f"        .in_dim2_pitch = {config['in_dim2_pitch']},\n")
            f.write(f"        .in_dim1_edge1 = {config['in_dim1_edge1']},\n")
            f.write(f"        .in_dim1_edge2 = {config['in_dim1_edge2']},\n")
            f.write(f"        .in_dim2_edge1 = {config['in_dim2_edge1']},\n")
            f.write(f"        .in_dim2_edge2 = {config['in_dim2_edge2']},\n")
            f.write(f"        .in_dim3_edge1 = {config['in_dim3_edge1']},\n")
            f.write(f"        .in_dim3_edge2 = {config['in_dim3_edge2']},\n")
            f.write(f"        .in_data_offset = {config['in_data_offset']},\n")
            f.write(f"        .in_rows_firstdma = {config['in_rows_firstdma']},\n")
            f.write(f"        \n")
            
            # Output tile dimensions
            f.write(f"        // Output tile: {config['out_dim1_size']}×{config['out_dim2_size']}×{config['out_dim3_size']}\n")
            f.write(f"        .out_dim1_size = {config['out_dim1_size']},\n")
            f.write(f"        .out_dim1_pitch = {config['out_dim1_pitch']},\n")
            f.write(f"        .out_dim2_size = {config['out_dim2_size']},\n")
            f.write(f"        .out_dim2_pitch = {config['out_dim2_pitch']},\n")
            f.write(f"        .out_dim3_size = {config['out_dim3_size']},\n")
            f.write(f"        \n")
            
            # Coefficient dimensions
            f.write(f"        // Coefficients: {config['coeff_dim1_size']}×{config['coeff_dim2_size']}×{config['coeff_dim3_size']}×{config['coeff_dim4_size']}\n")
            f.write(f"        .coeff_dim1_size = {config['coeff_dim1_size']},\n")
            f.write(f"        .coeff_dim2_size = {config['coeff_dim2_size']},\n")
            f.write(f"        .coeff_dim3_size = {config['coeff_dim3_size']},\n")
            f.write(f"        .coeff_dim4_size = {config['coeff_dim4_size']},\n")
            f.write(f"        .coeff_dim1_pitch = {config['coeff_dim1_pitch']},\n")
            f.write(f"        .coeff_dim2_pitch = {config['coeff_dim2_pitch']},\n")
            f.write(f"        .coeff_dim3_pitch = {config['coeff_dim3_pitch']},\n")
            f.write(f"        \n")
            
            # Bias and outscale
            f.write(f"        // Bias/Outscale: {config['bias_dim1_size']}\n")
            f.write(f"        .bias_dim1_size = {config['bias_dim1_size']},\n")
            f.write(f"        .bias_dim2_size = {config['bias_dim2_size']},\n")
            f.write(f"        .outscale_dim1_size = {config['outscale_dim1_size']},\n")
            f.write(f"        .outscale_dim2_size = {config['outscale_dim2_size']},\n")
            f.write(f"        \n")
            
            # Buffer sizes
            f.write(f"        // Buffer sizes (bytes)\n")
            f.write(f"        .input_buffer_size = {config['input_buffer_size']},\n")
            f.write(f"        .coeff_buffer_size = {config['coeff_buffer_size']},\n")
            f.write(f"        .output_buffer_size = {config['output_buffer_size']},\n")
            f.write(f"        .bias_buffer_size = {config['bias_buffer_size']},\n")
            f.write(f"        .outscale_buffer_size = {config['outscale_buffer_size']},\n")
            f.write(f"        \n")
            
            # DRAM placement
            f.write(f"        // DRAM placement\n")
            f.write(f"        .input_ping_dram = {config['input_ping_dram']},\n")
            f.write(f"        .input_pong_dram = {config['input_pong_dram']},\n")
            f.write(f"        .coeff_dram = {config['coeff_dram']},\n")
            f.write(f"        .output_ping_dram = {config['output_ping_dram']},\n")
            f.write(f"        .output_pong_dram = {config['output_pong_dram']},\n")
            f.write(f"        .bias_dram = {config['bias_dram']},\n")
            f.write(f"        .outscale_dram = {config['outscale_dram']},\n")
            f.write(f"        \n")
            
            # Tiling parameters
            f.write(f"        // Tiling: {config['n_tile_size']} ch/tile × {config['n_tiles']} tiles, {config['output_rows']} rows/tile × {config['height_tiles']} tiles\n")
            f.write(f"        .n_tile_size = {config['n_tile_size']},\n")
            f.write(f"        .n_tiles = {config['n_tiles']},\n")
            f.write(f"        .n_tile_size_last = {config['n_tile_size_last']},\n")
            f.write(f"        .height_tiles = {config['height_tiles']},\n")
            f.write(f"        .output_rows = {config['output_rows']},\n")
            f.write(f"        .input_rows = {config['input_rows']},\n")
            f.write(f"        \n")
            
            # Convolution parameters
            f.write(f"        // Conv params: {config['kernel_w']}×{config['kernel_h']}, stride {config['stride_x']}×{config['stride_y']}, pad {config['padding']}\n")
            f.write(f"        .kernel_w = {config['kernel_w']},\n")
            f.write(f"        .kernel_h = {config['kernel_h']},\n")
            f.write(f"        .stride_x = {config['stride_x']},\n")
            f.write(f"        .stride_y = {config['stride_y']},\n")
            f.write(f"        .padding = {config['padding']},\n")
            f.write(f"        .dilation = {config['dilation']},\n")
            f.write(f"        .accum_shift = {config['accum_shift']},\n")
            f.write(f"        .relu_max = {config['relu_max']},\n")
            f.write(f"        .relu_min = {config['relu_min']},\n")
            f.write(f"        .output_shift = {config['output_shift']},\n")
            f.write(f"        .output_scale = {config['output_scale']},\n")
            f.write(f"        .flags = {config['flags']},\n")
            f.write(f"        .input_zero_point = {config['input_zero_point']},\n")
            f.write("    },\n")
        
        f.write("};\n\n")
        
        # Generate accessor functions
        f.write("""
/**
 * Get total number of convolution layers
 */
static inline int get_num_conv_layers(void) {
    return NUM_CONV_LAYERS;
}

/**
 * Get configuration for a specific layer by layer_id
 * 
 * @param layer_id Layer index (0 to NUM_CONV_LAYERS-1)
 * @return Pointer to configuration, or NULL if invalid layer_id
 */
static inline const conv_layer_config_t* get_layer_config(int layer_id) {
    if (layer_id < 0 || layer_id >= NUM_CONV_LAYERS) {
        return NULL;
    }
    return &CONV_LAYER_CONFIGS[layer_id];
}

/**
 * Get configuration for a layer by its parameters
 * Searches for a layer matching the given convolution parameters
 *
 * @param ic   Input channels
 * @param ih   Input height
 * @param iw   Input width
 * @param oc   Output channels
 * @param kh   Kernel height
 * @param kw   Kernel width
 * @param oh   Output height
 * @param ow   Output width
 * @param sy   Stride Y
 * @param sx   Stride X
 * @param pad  Padding (symmetric)
 * @param dil  Dilation
 * @return Pointer to configuration, or NULL if not found
 */
static inline const conv_layer_config_t* get_layer_config_by_params(
    int ic, int ih, int iw,
    int oc, int kh, int kw,
    int oh, int ow,
    int sy, int sx,
    int pad, int dil) {
    
    for (int i = 0; i < NUM_CONV_LAYERS; i++) {
        const conv_layer_config_t* cfg = &CONV_LAYER_CONFIGS[i];
        if (cfg->src_dim3_size == ic &&
            cfg->src_dim2_size == ih &&
            cfg->src_dim1_size == iw &&
            cfg->dst_dim3_size == oc &&
            cfg->coeff_dim2_size == kh &&
            cfg->coeff_dim1_size == kw &&
            cfg->dst_dim2_size == oh &&
            cfg->dst_dim1_size == ow &&
            cfg->stride_y == sy &&
            cfg->stride_x == sx &&
            cfg->padding == pad &&
            cfg->dilation == dil) {
            return cfg;
        }
    }
    return NULL;
}

/**
 * Get configuration for a layer by config key string
 * Key format: "ic_ih_iw_oc_kh_kw_oh_ow_sy_sx_pad_dil"
 *
 * @param config_key The unique configuration key string
 * @return Pointer to configuration, or NULL if not found
 */
static inline const conv_layer_config_t* get_layer_config_by_key(const char* config_key) {
    if (config_key == NULL) return NULL;
    
    for (int i = 0; i < NUM_CONV_LAYERS; i++) {
        const conv_layer_config_t* cfg = &CONV_LAYER_CONFIGS[i];
        // Simple string comparison
        const char* a = cfg->config_key;
        const char* b = config_key;
        int match = 1;
        while (*a && *b) {
            if (*a++ != *b++) { match = 0; break; }
        }
        if (match && *a == *b) return cfg;
    }
    return NULL;
}

#endif // CONV_LAYER_CONFIGS_H
""")
    
    print(f"Generated {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate convolution layer configuration lookup table',
        epilog='One of --model, --pte, or a positional input_file (csv/json) is required.'
    )
    parser.add_argument('input_file', nargs='?', default=None,
                       help='Input file (layers_config.json or resnet_conv_list.csv). '
                            'Not needed when using --model or --pte.')
    parser.add_argument('--model', '-m', default=None,
                       help='Extract layers directly from PyTorch model(s). '
                            'Comma or + separated list. '
                            f'Supported: {", ".join(SUPPORTED_MODELS)}. '
                            'Example: --model resnet18+resnet50')
    parser.add_argument('--pte', nargs='+', default=None,
                       help='Extract layers from one or more ExecuTorch .pte binaries. '
                            'Example: --pte resnet18.pte resnet50.pte')
    parser.add_argument('--flatc', default=None,
                       help='Path to flatc binary (default: cmake-out/third-party/flatc_ep/bin/flatc)')
    parser.add_argument('--input-size', default='1,3,64,64',
                       help='Model input tensor shape as N,C,H,W (default: 1,3,64,64)')
    parser.add_argument('--output', '-o', default='conv_layer_configs.h',
                       help='Output C header file (default: conv_layer_configs.h)')
    parser.add_argument('--dram0', type=int, default=DRAM_SIZE_0,
                       help=f'DRAM0 size in bytes (default: {DRAM_SIZE_0})')
    parser.add_argument('--dram1', type=int, default=DRAM_SIZE_1,
                       help=f'DRAM1 size in bytes (default: {DRAM_SIZE_1})')
    parser.add_argument('--no-dma-mode', action='store_true', default=False,
                       help='Force all configs to no-DMA mode: changes _dma suffix to _no_dma for every kernel name')
    
    args = parser.parse_args()
    
    # ---- Load layers: --model, --pte, or input_file ----
    if args.pte:
        all_layers = []
        seen_keys = set()
        for pte_arg in args.pte:
            pte_path = Path(pte_arg)
            if not pte_path.exists():
                print(f"ERROR: PTE file not found: {pte_path}")
                return 1
            print(f"Extracting layers from PTE: {pte_path}")
            pte_layers = load_layers_from_pte(pte_path, flatc_path=args.flatc)
            for l in pte_layers:
                key = (l['input'], l['output'], l['kernel'],
                       l['stride'], l['padding'], l['dilation'])
                if key not in seen_keys:
                    seen_keys.add(key)
                    l['layer_id'] = len(all_layers)
                    all_layers.append(l)
                else:
                    print(f"  [skip duplicate] {l['name']}")
        layers = all_layers
        print(f"Total unique layers from {len(args.pte)} PTE file(s): {len(layers)}")
    elif args.model:
        # Parse model names (accept comma or + as separator)
        model_names = [n.strip() for n in args.model.replace('+', ',').split(',') if n.strip()]
        input_size = tuple(int(x) for x in args.input_size.split(','))
        print(f"Extracting layers from model(s): {', '.join(model_names)}  input_size={input_size}")
        layers = load_layers_from_model(model_names, input_size)
    elif args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"ERROR: Input file not found: {input_path}")
            return 1
        print(f"Loading layers from {input_path}...")
        if input_path.suffix == '.json':
            layers = load_layers_from_json(input_path)
        elif input_path.suffix == '.csv':
            layers = load_layers_from_csv(input_path)
        else:
            print(f"ERROR: Unsupported file type: {input_path.suffix}")
            print("Supported: .json, .csv")
            return 1
    else:
        parser.error('One of --model, --pte, or a positional input_file is required.')
        return 1
    
    print(f"Loaded {len(layers)} layers")
    
    # Calculate configurations for all layers
    print(f"\nCalculating buffer configurations (DRAM0={args.dram0}, DRAM1={args.dram1})...")
    configs = []
    for layer in layers:
        print(f"  Processing layer {layer['layer_id']}: {layer['name']}...")
        config = calculate_layer_config(layer, args.dram0, args.dram1)
        if config:
            configs.append(config)
            print(f"    [OK] n_tile={config['n_tile_size']}, n_tiles={config['n_tiles']}, "
                  f"output_rows={config['output_rows']}, height_tiles={config['height_tiles']}")
        else:
            print(f"    ✗ Failed to calculate configuration")
    
    if len(configs) == 0:
        print("ERROR: No valid configurations generated")
        return 1
    
    print(f"\nGenerated {len(configs)} valid configurations")
    
    # Apply no-DMA mode: change _dma suffix to _no_dma for every kernel name
    if args.no_dma_mode:
        for config in configs:
            if config['kernel_name'].endswith('_dma'):
                config['kernel_name'] = config['kernel_name'][:-4] + '_no_dma'
        print(f"No-DMA mode enabled: all kernel names suffixed with _no_dma")
    
    # Generate C header
    generate_c_header(configs, args.output, args.dram0, args.dram1, args.no_dma_mode)
    
    print(f"\nSuccess! Generated {args.output}")
    print(f"Use in C code:")
    print(f"  #include \"{args.output}\"")
    print(f"  const conv_layer_config_t* config = get_layer_config(0);")
    print(f"  conv_execute_layer(0, input, output, weights, bias, outscale);")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
