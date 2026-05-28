#!/usr/bin/env python3
"""
Generate combined conv2d + maxpool DMA buffer configuration header from PTE files.

Extracts both conv2d and maxpool layers from ExecuTorch .pte binaries and
generates a single C header with both configuration tables and accessors.

Usage:
    # Single PTE
    python generate_combined_configs.py \\
        --pte resnet18_quantized.pte \\
        --output layer_configs.h --dram0 62976 --dram1 62976

    # Multiple PTE files (deduplicates automatically)
    python generate_combined_configs.py \\
        --pte resnet18_quantized.pte resnet50_quantized.pte \\
        --output layer_configs.h --dram0 62976 --dram1 62976

    # Force all conv kernels to no-DMA mode
    python generate_combined_configs.py \\
        --pte resnet18_quantized.pte \\
        --output layer_configs.h --dram0 62976 --dram1 62976 --no-dma-mode
"""

import os
import sys
import json
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXECUTORCH_ROOT = _SCRIPT_DIR.parents[3]          # backends/cadence/vision/config_generator -> executorch/
_EXECUTORCH_SRC  = str(_EXECUTORCH_ROOT / 'src' / 'executorch')
_EXECUTORCH_PARENT = str(_EXECUTORCH_ROOT / 'src')

# Try multiple known flatc locations
_FLATC_CANDIDATES = [
    _EXECUTORCH_ROOT / 'cmake-out' / 'third-party' / 'flatc_ep' / 'bin' / 'flatc',
    _EXECUTORCH_ROOT / 'cmake-out-generic-all' / 'third-party' / 'flatc_ep' / 'bin' / 'flatc',
    _EXECUTORCH_ROOT / 'pip-out' / 'lib.linux-x86_64-cpython-311' / 'executorch' / 'data' / 'bin' / 'flatc',
    _EXECUTORCH_ROOT / 'pip-out' / 'temp.linux-x86_64-cpython-311' / 'cmake-out' / 'third-party' / 'flatc_ep' / 'bin' / 'flatc',
]
_FLATC_DEFAULT = str(next((p for p in _FLATC_CANDIDATES if p.exists()), _FLATC_CANDIDATES[0]))

# Import conv buffer calculation
sys.path.insert(0, str(_SCRIPT_DIR))
from generate_idma_buffers import (
    find_max_tile_config,
    calculate_buffer_sizes_with_rows,
    calculate_buffer_placement,
    DRAM_SIZE_0,
    DRAM_SIZE_1,
)

ELEMENT_SIZE_F32 = 4  # float32 bytes


# =====================================================================
# Bootstrap executorch imports (shared with generate_layer_configs.py)
# =====================================================================

def _bootstrap_executorch_imports(flatc_path=None):
    import types
    if _EXECUTORCH_PARENT not in sys.path:
        sys.path.insert(0, _EXECUTORCH_PARENT)
    if _EXECUTORCH_SRC not in sys.path:
        sys.path.insert(0, _EXECUTORCH_SRC)
    for pkg, pkg_dir in [
        ('executorch',       _EXECUTORCH_SRC),
        ('executorch.exir',  _EXECUTORCH_SRC + '/exir'),
    ]:
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [pkg_dir]
            m.__package__ = pkg
            sys.modules[pkg] = m
    resolved = flatc_path or _FLATC_DEFAULT
    if os.path.isfile(resolved):
        os.environ.setdefault('FLATC_EXECUTABLE', resolved)


# =====================================================================
# PTE extraction — conv2d and maxpool
# =====================================================================

def extract_layers_from_pte(pte_file, flatc_path=None):
    """
    Extract conv2d and maxpool layers from a .pte binary.

    Returns:
        (conv_layers, maxpool_layers)
        Each is a list of dicts in the internal format.
    """
    _bootstrap_executorch_imports(flatc_path)

    from executorch.exir._serialize._program import deserialize_pte_binary
    from executorch.exir.schema import KernelCall, Int, IntList, Tensor

    pte_path = Path(pte_file)
    print(f"Loading PTE: {pte_path} ...")

    with open(pte_path, 'rb') as f:
        pte_file_obj = deserialize_pte_binary(f.read())

    if hasattr(pte_file_obj, 'program'):
        program = pte_file_obj.program
    else:
        program = pte_file_obj

    plan   = program.execution_plan[0]
    values = plan.values

    def _tensor(idx):
        v = values[idx].val
        return v if isinstance(v, Tensor) else None

    def _int_val(idx):
        v = values[idx].val
        return v.int_val if isinstance(v, Int) else None

    def _intlist_val(idx):
        v = values[idx].val
        if isinstance(v, IntList):
            return [_int_val(i) for i in v.items]
        return None

    CONV_OPS = {
        'cadence::quantized_conv2d_nchw',
        'aten::conv2d',
        'aten::convolution',
    }
    MAXPOOL_OPS = {
        'aten::max_pool2d_with_indices',
        'aten::max_pool2d',
    }

    conv_layers = []
    conv_seen = set()
    maxpool_layers = []
    maxpool_seen = set()

    for instr in plan.chains[0].instructions:
        ia = instr.instr_args
        if not isinstance(ia, KernelCall):
            continue
        op_name = plan.operators[ia.op_index].name
        args = ia.args

        # --- Conv2d ---
        if op_name in CONV_OPS:
            input_t  = _tensor(args[0])
            weight_t = _tensor(args[1])
            output_t = _tensor(args[-1])
            if input_t is None or weight_t is None or output_t is None:
                continue

            stride   = _intlist_val(args[3]) or [1, 1]
            padding  = _intlist_val(args[4]) or [0, 0]
            dilation = _intlist_val(args[5]) or [1, 1]

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
            if key not in conv_seen:
                conv_seen.add(key)
                conv_layers.append(info)

        # --- MaxPool ---
        elif op_name in MAXPOOL_OPS:
            input_t  = _tensor(args[0])
            # max_pool2d_with_indices: input, kernel_size, stride, padding, dilation, ceil_mode, output, indices
            # max_pool2d:              input, kernel_size, stride, padding, dilation, ceil_mode, output
            if input_t is None:
                continue

            kernel_size = _intlist_val(args[1]) or [2, 2]
            mp_stride   = _intlist_val(args[2]) or kernel_size
            mp_padding  = _intlist_val(args[3]) or [0, 0]

            _, C, H, W = input_t.sizes
            kh, kw = kernel_size[0], kernel_size[1]
            sh, sw = mp_stride[0], mp_stride[1]
            ph, pw = mp_padding[0], mp_padding[1]

            mp_key = (C, H, W, kh, kw, sh, sw, ph, pw)
            if mp_key not in maxpool_seen:
                maxpool_seen.add(mp_key)
                maxpool_layers.append({
                    'name': f"maxpool_{kh}x{kw}s{sh}_c{C}_{H}x{W}",
                    'src_width': W,
                    'src_height': H,
                    'channels': C,
                    'kernel_h': kh,
                    'kernel_w': kw,
                    'stride_h': sh,
                    'stride_w': sw,
                    'pad_h': ph,
                    'pad_w': pw,
                })

    # Convert conv to internal format
    conv_result = []
    for layer_id, info in enumerate(conv_layers):
        in_w, in_h, in_c = info['input']
        out_w, out_h, out_c = info['output']
        k_w, k_h, _ic, _oc = info['kernel']
        name = f"conv_{k_h}x{k_w}_s{info['stride'][0]}_ic{in_c}_oc{out_c}"
        conv_result.append({
            'layer_id': layer_id,
            'name': name,
            'input': info['input'],
            'output': info['output'],
            'kernel': info['kernel'],
            'stride': info['stride'],
            'padding': info['padding'],
            'dilation': info['dilation'],
        })

    print(f"  Extracted {len(conv_result)} conv layers, {len(maxpool_layers)} maxpool layers")
    return conv_result, maxpool_layers


# =====================================================================
# Conv config calculation (reused from generate_layer_configs.py)
# =====================================================================

def calculate_conv_config(layer, dram0_size, dram1_size):
    """Calculate complete conv config dict for one layer.
    Mirrors calculate_layer_config() in generate_layer_configs.py."""
    in_w, in_h, in_c = layer['input']
    out_w, out_h, out_c = layer['output']
    k_w, k_h, _, _ = layer['kernel']
    stride_w, stride_h = layer['stride']
    pad = layer['padding'][0]
    dil = layer['dilation'][0]
    pad_w = pad_h = pad

    padding = (pad_w, pad_w, pad_h, pad_h, 0, 0)
    conv_params = (stride_w, stride_h, 8, 4000, 11, 0, 1, k_h, k_w)

    # Kernel name
    if k_h == 7 and k_w == 7 and stride_h == 2:
        kernel_name = "7x7j2d1"
    elif k_h == 3 and k_w == 3 and stride_h == 1:
        kernel_name = "3x3j1d1"
    elif k_h == 3 and k_w == 3 and stride_h == 2:
        kernel_name = "3x3j2d1"
    elif k_h == 1 and k_w == 1 and stride_h == 2:
        kernel_name = "1x1j2d1"
    elif k_h == 1 and k_w == 1 and stride_h == 1:
        kernel_name = "1x1j1d1"
    else:
        kernel_name = f"{k_w}x{k_h}j{stride_w}d1"

    n_tile_size, output_rows, buffer_sizes = find_max_tile_config(
        input_whd=(in_w, in_h, in_c),
        output_whd=(out_w, out_h, out_c),
        kernel_whdn=(k_w, k_h, in_c, out_c),
        padding=padding,
        stride_xy=(stride_w, stride_h),
        kernel_name=kernel_name,
        data_type="S8S8",
        dram0_size=dram0_size,
        dram1_size=dram1_size,
        conv_params=conv_params,
    )

    if buffer_sizes is None or n_tile_size == 0 or output_rows == 0:
        # No-DMA fallback
        in_dim1_pitch = in_w + 2 * pad_w
        in_dim2_pitch = in_dim1_pitch * (in_h + 2 * pad_h)
        out_dim1_pitch = out_w
        out_dim2_pitch = out_dim1_pitch * out_h
        coeff_dim1_pitch = k_w
        coeff_dim2_pitch = coeff_dim1_pitch * k_h
        coeff_dim3_pitch = coeff_dim2_pitch * in_c

        return {
            'layer_id': layer['layer_id'], 'layer_name': layer['name'],
            'kernel_name': kernel_name + "_no_dma",
            'src_dim1_size': in_w, 'src_dim2_size': in_h, 'src_dim3_size': in_c,
            'src_dim1_pitch': in_w, 'src_dim2_pitch': in_w * in_h,
            'dst_dim1_size': out_w, 'dst_dim2_size': out_h, 'dst_dim3_size': out_c,
            'dst_dim1_pitch': out_w, 'dst_dim2_pitch': out_w * out_h,
            'in_dim1_size': in_w, 'in_dim1_pitch': in_dim1_pitch,
            'in_dim2_size': in_h, 'in_dim2_pitch': in_dim2_pitch,
            'in_dim1_edge1': pad_w, 'in_dim1_edge2': pad_w,
            'in_dim2_edge1': pad_h, 'in_dim2_edge2': pad_h,
            'in_dim3_edge1': 0, 'in_dim3_edge2': 0,
            'in_data_offset': 0, 'in_rows_firstdma': in_h,
            'out_dim1_size': out_w, 'out_dim1_pitch': out_dim1_pitch,
            'out_dim2_size': out_h, 'out_dim2_pitch': out_dim2_pitch,
            'out_dim3_size': out_c,
            'coeff_dim1_size': k_w, 'coeff_dim2_size': k_h,
            'coeff_dim3_size': in_c, 'coeff_dim4_size': out_c,
            'coeff_dim1_pitch': coeff_dim1_pitch, 'coeff_dim2_pitch': coeff_dim2_pitch,
            'coeff_dim3_pitch': coeff_dim3_pitch,
            'bias_dim1_size': out_c, 'bias_dim2_size': 1,
            'outscale_dim1_size': out_c, 'outscale_dim2_size': 1,
            'input_buffer_size': in_dim2_pitch * in_c,
            'coeff_buffer_size': coeff_dim3_pitch * out_c,
            'output_buffer_size': out_dim2_pitch * out_c,
            'bias_buffer_size': out_c * 4, 'outscale_buffer_size': out_c * 2,
            'input_ping_dram': 0, 'input_pong_dram': 0, 'coeff_dram': 0,
            'output_ping_dram': 0, 'output_pong_dram': 0,
            'bias_dram': 0, 'outscale_dram': 0,
            'n_tile_size': out_c, 'n_tiles': 1, 'n_tile_size_last': out_c,
            'height_tiles': 1, 'output_rows': out_h, 'input_rows': in_h,
            'kernel_w': k_w, 'kernel_h': k_h,
            'stride_x': stride_w, 'stride_y': stride_h,
            'padding': pad_w, 'dilation': 1,
            'accum_shift': 8, 'relu_max': 4000, 'relu_min': 0,
            'output_shift': 11, 'output_scale': 0, 'flags': 0,
            'input_zero_point': 0,
            'config_key': f"{in_c}_{in_h}_{in_w}_{out_c}_{k_h}_{k_w}_{out_h}_{out_w}_{stride_h}_{stride_w}_{pad_w}_1",
        }

    # DMA mode — use buffer_sizes dict from find_max_tile_config
    n_tiles = (out_c + n_tile_size - 1) // n_tile_size
    height_tiles = (out_h + output_rows - 1) // output_rows
    input_rows = k_h + (output_rows - 1) * stride_h

    placement = calculate_buffer_placement(buffer_sizes, dram0_size, dram1_size)

    dilation = buffer_sizes.get('DILATION', 1)
    config = {
        'layer_id': layer['layer_id'], 'layer_name': layer['name'],
        'kernel_name': kernel_name + "_dma",
        'src_dim1_size': buffer_sizes['SRC_DIM1_SIZE'],
        'src_dim2_size': buffer_sizes['SRC_DIM2_SIZE'],
        'src_dim3_size': buffer_sizes['SRC_DIM3_SIZE'],
        'src_dim1_pitch': buffer_sizes['SRC_DIM1_PITCH'],
        'src_dim2_pitch': buffer_sizes['SRC_DIM2_PITCH'],
        'dst_dim1_size': buffer_sizes['DST_DIM1_SIZE'],
        'dst_dim2_size': buffer_sizes['DST_DIM2_SIZE'],
        'dst_dim3_size': out_c,
        'dst_dim1_pitch': buffer_sizes['DST_DIM1_PITCH'],
        'dst_dim2_pitch': buffer_sizes['DST_DIM2_PITCH'],
        'in_dim1_size': buffer_sizes['IN_DIM1_SIZE'],
        'in_dim1_pitch': buffer_sizes['IN_DIM1_PITCH'],
        'in_dim2_size': buffer_sizes['IN_DIM2_SIZE'],
        'in_dim2_pitch': buffer_sizes['IN_DIM2_PITCH'],
        'in_dim1_edge1': padding[0], 'in_dim1_edge2': padding[1],
        'in_dim2_edge1': padding[2], 'in_dim2_edge2': padding[3],
        'in_dim3_edge1': padding[4], 'in_dim3_edge2': padding[5],
        'in_data_offset': buffer_sizes['IN_DATA_OFFSET'],
        'in_rows_firstdma': buffer_sizes['IN_ROWS_FIRSTDMA'],
        'out_dim1_size': buffer_sizes['OUT_DIM1_SIZE'],
        'out_dim1_pitch': buffer_sizes['OUT_DIM1_PITCH'],
        'out_dim2_size': buffer_sizes['OUT_DIM2_SIZE'],
        'out_dim2_pitch': buffer_sizes['OUT_DIM2_PITCH'],
        'out_dim3_size': buffer_sizes['OUT_DIM3_SIZE'],
        'coeff_dim1_size': buffer_sizes['COEFF_DIM1_SIZE'],
        'coeff_dim2_size': buffer_sizes['COEFF_DIM2_SIZE'],
        'coeff_dim3_size': buffer_sizes['COEFF_DIM3_SIZE'],
        'coeff_dim4_size': buffer_sizes['COEFF_DIM4_SIZE'],
        'coeff_dim1_pitch': buffer_sizes['COEFF_DIM1_PITCH'],
        'coeff_dim2_pitch': buffer_sizes['COEFF_DIM2_PITCH'],
        'coeff_dim3_pitch': buffer_sizes['COEFF_DIM3_PITCH'],
        'bias_dim1_size': buffer_sizes['BIAS_DIM1_SIZE'],
        'bias_dim2_size': buffer_sizes['BIAS_DIM2_SIZE'],
        'outscale_dim1_size': buffer_sizes['OUTSCALE_DIM1_SIZE'],
        'outscale_dim2_size': buffer_sizes['OUTSCALE_DIM2_SIZE'],
        'input_buffer_size': buffer_sizes['IN'],
        'coeff_buffer_size': buffer_sizes['COEFF'],
        'output_buffer_size': buffer_sizes['OUT'],
        'bias_buffer_size': buffer_sizes['BIAS'],
        'outscale_buffer_size': buffer_sizes['OUTSCALE'],
        'input_ping_dram': placement.get('IN1_dram', 0),
        'input_pong_dram': placement.get('IN2_dram', 1),
        'coeff_dram': placement.get('COEFF_dram', 0),
        'output_ping_dram': placement.get('OUT1_dram', 1),
        'output_pong_dram': placement.get('OUT2_dram', 1),
        'bias_dram': placement.get('BIAS_dram', 1),
        'outscale_dram': placement.get('OUTSCALE_dram', 1),
        'n_tile_size': buffer_sizes['N_TILE_SIZE'],
        'n_tiles': buffer_sizes['N_TILES'],
        'n_tile_size_last': buffer_sizes['N_TILE_SIZE_LAST'],
        'height_tiles': buffer_sizes['HIGHT_TILES'],
        'output_rows': output_rows,
        'input_rows': input_rows,
        'stride_x': buffer_sizes.get('STRIDEX', stride_w),
        'stride_y': buffer_sizes.get('STRIDEY', stride_h),
        'accum_shift': buffer_sizes.get('ACCUM_SHIFT', 8),
        'relu_max': buffer_sizes.get('RELU_MAX', 4000),
        'relu_min': buffer_sizes.get('RELU_MIN', 0),
        'output_shift': buffer_sizes.get('OUTPUT_SHIFT', 11),
        'output_scale': buffer_sizes.get('OUTPUT_SCALE', 0),
        'dilation': dilation,
        'kernel_w': k_w, 'kernel_h': k_h,
        'padding': pad_w, 'flags': buffer_sizes.get('FLAGS', 0),
        'input_zero_point': 0,
        'config_key': f"{in_c}_{in_h}_{in_w}_{out_c}_{k_h}_{k_w}_{out_h}_{out_w}_{stride_h}_{stride_w}_{pad_w}_{dilation}",
    }
    return config


# =====================================================================
# Maxpool config calculation (reused from generate_maxpool_configs.py)
# =====================================================================

def calculate_maxpool_buffers(layer, c_tile_size, output_rows):
    W  = layer['src_width']
    H  = layer['src_height']
    kh = layer['kernel_h']
    kw = layer['kernel_w']
    sh = layer['stride_h']
    sw = layer['stride_w']
    ph = layer['pad_h']
    pw = layer['pad_w']

    dst_w = (W + 2 * pw - kw) // sw + 1
    dst_h = (H + 2 * ph - kh) // sh + 1

    input_rows   = (output_rows - 1) * sh + kh
    in_tile_w    = W + 2 * pw
    in_tile_rows = input_rows + 2 * ph
    in_tile_plane = in_tile_w * in_tile_rows
    in_data_offset = ph * in_tile_w + pw

    out_tile_plane = dst_w * output_rows
    input_buf  = c_tile_size * in_tile_plane * ELEMENT_SIZE_F32
    output_buf = c_tile_size * out_tile_plane * ELEMENT_SIZE_F32

    C = layer['channels']
    c_tiles = (C + c_tile_size - 1) // c_tile_size
    c_tile_last = C - c_tile_size * (c_tiles - 1)
    height_tiles = (dst_h + output_rows - 1) // output_rows

    return {
        'dst_width': dst_w, 'dst_height': dst_h,
        'input_rows': input_rows,
        'in_tile_w': in_tile_w, 'in_tile_rows': in_tile_rows,
        'in_tile_plane': in_tile_plane, 'in_data_offset': in_data_offset,
        'out_tile_w': dst_w, 'out_tile_rows': output_rows,
        'out_tile_plane': out_tile_plane,
        'c_tile_size': c_tile_size, 'c_tiles': c_tiles,
        'c_tile_size_last': c_tile_last,
        'height_tiles': height_tiles, 'output_rows': output_rows,
        'input_buffer_size': input_buf, 'output_buffer_size': output_buf,
    }


def find_maxpool_tiling(layer, dram0_size, dram1_size):
    C = layer['channels']
    dst_h = ((layer['src_height'] + 2 * layer['pad_h'] - layer['kernel_h'])
             // layer['stride_h'] + 1)
    bank = min(dram0_size, dram1_size)

    best_c, best_r, best_buf = 0, 0, None
    for c in range(C, 0, -1):
        for r in range(dst_h, 0, -1):
            buf = calculate_maxpool_buffers(layer, c, r)
            if buf['input_buffer_size'] + buf['output_buffer_size'] <= bank:
                if (c > best_c) or (c == best_c and r > best_r):
                    best_c, best_r, best_buf = c, r, buf
                break
        if best_c == C:
            break
    return best_c, best_r, best_buf


def build_maxpool_config(layer_id, layer, dram0_size, dram1_size):
    c_tile, out_rows, buf = find_maxpool_tiling(layer, dram0_size, dram1_size)
    if buf is None:
        dst_h = ((layer['src_height'] + 2 * layer['pad_h'] - layer['kernel_h'])
                 // layer['stride_h'] + 1)
        c_tile = layer['channels']
        out_rows = dst_h
        buf = calculate_maxpool_buffers(layer, c_tile, out_rows)

    W = layer['src_width']
    H = layer['src_height']
    C = layer['channels']

    cfg = {
        'layer_id': layer_id,
        'layer_name': layer.get('name', f"maxpool_{layer_id}"),
        'config_key': f"{C}_{H}_{W}_{layer['kernel_h']}_{layer['kernel_w']}_"
                      f"{layer['stride_h']}_{layer['stride_w']}_"
                      f"{layer['pad_h']}_{layer['pad_w']}",
        'src_width': W, 'src_height': H, 'channels': C,
        'dst_width': buf['dst_width'], 'dst_height': buf['dst_height'],
        'src_row_pitch': W, 'src_plane_pitch': H * W,
        'dst_row_pitch': buf['dst_width'],
        'dst_plane_pitch': buf['dst_height'] * buf['dst_width'],
        'kernel_h': layer['kernel_h'], 'kernel_w': layer['kernel_w'],
        'stride_h': layer['stride_h'], 'stride_w': layer['stride_w'],
        'pad_h': layer['pad_h'], 'pad_w': layer['pad_w'],
        'in_tile_w': buf['in_tile_w'], 'in_tile_rows': buf['in_tile_rows'],
        'in_tile_plane': buf['in_tile_plane'],
        'in_data_offset': buf['in_data_offset'],
        'out_tile_w': buf['out_tile_w'], 'out_tile_rows': buf['out_tile_rows'],
        'out_tile_plane': buf['out_tile_plane'],
        'c_tile_size': buf['c_tile_size'], 'c_tiles': buf['c_tiles'],
        'c_tile_size_last': buf['c_tile_size_last'],
        'height_tiles': buf['height_tiles'],
        'output_rows': buf['output_rows'], 'input_rows': buf['input_rows'],
        'input_buffer_size': buf['input_buffer_size'],
        'output_buffer_size': buf['output_buffer_size'],
        'input_ping_dram': 0, 'input_pong_dram': 1,
        'output_ping_dram': 1, 'output_pong_dram': 0,
    }
    return cfg


# =====================================================================
# Combined C header generation
# =====================================================================

def generate_combined_header(conv_configs, maxpool_configs, output_file,
                             dram0_size, dram1_size, no_dma_mode=False):
    _dram0 = 0 if no_dma_mode else dram0_size
    _dram1 = 0 if no_dma_mode else dram1_size

    with open(output_file, 'w') as f:
        f.write("""\
/*
 * layer_configs.h
 *
 * Auto-generated conv2d + maxpool layer configurations
 * Generated from PTE extraction by generate_combined_configs.py
 *
 * DO NOT EDIT MANUALLY
 */

#ifndef LAYER_CONFIGS_H
#define LAYER_CONFIGS_H

#include <stdint.h>
#include <stddef.h>  /* for NULL */

""")
        # ----------------------------------------------------------
        # DRAM macros
        # ----------------------------------------------------------
        f.write(f"#define IDMA_BUFFER_SIZE_DRAM0 ({_dram0})  /* {_dram0 // 1024} KB */\n")
        f.write(f"#define IDMA_BUFFER_SIZE_DRAM1 ({_dram1})  /* {_dram1 // 1024} KB */\n\n")

        # ===========================================================
        # CONV SECTION
        # ===========================================================
        f.write("/* " + "=" * 70 + " */\n")
        f.write("/*  Conv2d configurations                                              */\n")
        f.write("/* " + "=" * 70 + " */\n\n")

        f.write("""\
typedef struct {
    int layer_id;
    const char* layer_name;
    const char* kernel_name;
    const char* config_key;

    int src_dim1_size;  int src_dim2_size;  int src_dim3_size;
    int src_dim1_pitch; int src_dim2_pitch;

    int dst_dim1_size;  int dst_dim2_size;  int dst_dim3_size;
    int dst_dim1_pitch; int dst_dim2_pitch;

    int in_dim1_size;   int in_dim1_pitch;
    int in_dim2_size;   int in_dim2_pitch;
    int in_dim1_edge1;  int in_dim1_edge2;
    int in_dim2_edge1;  int in_dim2_edge2;
    int in_dim3_edge1;  int in_dim3_edge2;
    int in_data_offset; int in_rows_firstdma;

    int out_dim1_size;  int out_dim1_pitch;
    int out_dim2_size;  int out_dim2_pitch;
    int out_dim3_size;

    int coeff_dim1_size;  int coeff_dim2_size;
    int coeff_dim3_size;  int coeff_dim4_size;
    int coeff_dim1_pitch; int coeff_dim2_pitch; int coeff_dim3_pitch;

    int bias_dim1_size;     int bias_dim2_size;
    int outscale_dim1_size; int outscale_dim2_size;

    int input_buffer_size;  int coeff_buffer_size;  int output_buffer_size;
    int bias_buffer_size;   int outscale_buffer_size;

    int input_ping_dram;  int input_pong_dram;  int coeff_dram;
    int output_ping_dram; int output_pong_dram;
    int bias_dram;        int outscale_dram;

    int n_tile_size; int n_tiles; int n_tile_size_last;
    int height_tiles; int output_rows; int input_rows;

    int kernel_w; int kernel_h;
    int stride_x; int stride_y;
    int padding;  int dilation;
    int accum_shift; int relu_max; int relu_min;
    int output_shift; int output_scale; int flags;
    int input_zero_point;
} conv_layer_config_t;

""")
        f.write(f"#define NUM_CONV_LAYERS {len(conv_configs)}\n\n")
        f.write("static const conv_layer_config_t CONV_LAYER_CONFIGS[] = {\n")

        conv_fields = [
            'layer_id', 'layer_name', 'kernel_name', 'config_key',
            'src_dim1_size', 'src_dim2_size', 'src_dim3_size', 'src_dim1_pitch', 'src_dim2_pitch',
            'dst_dim1_size', 'dst_dim2_size', 'dst_dim3_size', 'dst_dim1_pitch', 'dst_dim2_pitch',
            'in_dim1_size', 'in_dim1_pitch', 'in_dim2_size', 'in_dim2_pitch',
            'in_dim1_edge1', 'in_dim1_edge2', 'in_dim2_edge1', 'in_dim2_edge2',
            'in_dim3_edge1', 'in_dim3_edge2', 'in_data_offset', 'in_rows_firstdma',
            'out_dim1_size', 'out_dim1_pitch', 'out_dim2_size', 'out_dim2_pitch', 'out_dim3_size',
            'coeff_dim1_size', 'coeff_dim2_size', 'coeff_dim3_size', 'coeff_dim4_size',
            'coeff_dim1_pitch', 'coeff_dim2_pitch', 'coeff_dim3_pitch',
            'bias_dim1_size', 'bias_dim2_size', 'outscale_dim1_size', 'outscale_dim2_size',
            'input_buffer_size', 'coeff_buffer_size', 'output_buffer_size',
            'bias_buffer_size', 'outscale_buffer_size',
            'input_ping_dram', 'input_pong_dram', 'coeff_dram',
            'output_ping_dram', 'output_pong_dram', 'bias_dram', 'outscale_dram',
            'n_tile_size', 'n_tiles', 'n_tile_size_last',
            'height_tiles', 'output_rows', 'input_rows',
            'kernel_w', 'kernel_h', 'stride_x', 'stride_y', 'padding', 'dilation',
            'accum_shift', 'relu_max', 'relu_min', 'output_shift', 'output_scale', 'flags',
            'input_zero_point',
        ]
        str_fields = {'layer_name', 'kernel_name', 'config_key'}

        for cfg in conv_configs:
            f.write("    {\n")
            for fld in conv_fields:
                val = cfg[fld]
                if fld in str_fields:
                    f.write(f"        .{fld} = \"{val}\",\n")
                else:
                    f.write(f"        .{fld} = {val},\n")
            f.write("    },\n")
        f.write("};\n\n")

        # Conv accessors
        f.write("""\
static inline int get_num_conv_layers(void) { return NUM_CONV_LAYERS; }

static inline const conv_layer_config_t* get_conv_config(int layer_id) {
    if (layer_id < 0 || layer_id >= NUM_CONV_LAYERS) return NULL;
    return &CONV_LAYER_CONFIGS[layer_id];
}

static inline const conv_layer_config_t* get_layer_config_by_params(
    int ic, int ih, int iw,
    int oc, int kh, int kw,
    int oh, int ow,
    int sy, int sx,
    int pad, int dil)
{
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
            cfg->dilation == dil)
            return cfg;
    }
    return NULL;
}

static inline const conv_layer_config_t* get_layer_config_by_key(const char* config_key) {
    if (config_key == NULL) return NULL;
    for (int i = 0; i < NUM_CONV_LAYERS; i++) {
        const conv_layer_config_t* cfg = &CONV_LAYER_CONFIGS[i];
        if (cfg->config_key != NULL) {
            const char* a = config_key;
            const char* b = cfg->config_key;
            while (*a && *b && *a == *b) { a++; b++; }
            if (*a == '\\0' && *b == '\\0') return cfg;
        }
    }
    return NULL;
}

""")

        # ===========================================================
        # MAXPOOL SECTION
        # ===========================================================
        f.write("/* " + "=" * 70 + " */\n")
        f.write("/*  MaxPool configurations                                             */\n")
        f.write("/* " + "=" * 70 + " */\n\n")

        f.write("""\
typedef struct {
    int layer_id;
    const char* layer_name;
    const char* config_key;

    int src_width;  int src_height;  int channels;
    int dst_width;  int dst_height;

    int src_row_pitch;  int src_plane_pitch;
    int dst_row_pitch;  int dst_plane_pitch;

    int kernel_h;  int kernel_w;
    int stride_h;  int stride_w;
    int pad_h;     int pad_w;

    int in_tile_w;      int in_tile_rows;   int in_tile_plane;
    int in_data_offset;
    int out_tile_w;     int out_tile_rows;  int out_tile_plane;

    int c_tile_size;  int c_tiles;  int c_tile_size_last;
    int height_tiles; int output_rows; int input_rows;

    int input_buffer_size;  int output_buffer_size;

    int input_ping_dram;  int input_pong_dram;
    int output_ping_dram; int output_pong_dram;
} maxpool_layer_config_t;

""")
        f.write(f"#define NUM_MAXPOOL_LAYERS {len(maxpool_configs)}\n\n")
        f.write("static const maxpool_layer_config_t MAXPOOL_LAYER_CONFIGS[] = {\n")

        mp_fields = [
            'layer_id', 'layer_name', 'config_key',
            'src_width', 'src_height', 'channels',
            'dst_width', 'dst_height',
            'src_row_pitch', 'src_plane_pitch', 'dst_row_pitch', 'dst_plane_pitch',
            'kernel_h', 'kernel_w', 'stride_h', 'stride_w', 'pad_h', 'pad_w',
            'in_tile_w', 'in_tile_rows', 'in_tile_plane', 'in_data_offset',
            'out_tile_w', 'out_tile_rows', 'out_tile_plane',
            'c_tile_size', 'c_tiles', 'c_tile_size_last',
            'height_tiles', 'output_rows', 'input_rows',
            'input_buffer_size', 'output_buffer_size',
            'input_ping_dram', 'input_pong_dram', 'output_ping_dram', 'output_pong_dram',
        ]
        mp_str_fields = {'layer_name', 'config_key'}

        for cfg in maxpool_configs:
            f.write("    {\n")
            for fld in mp_fields:
                val = cfg[fld]
                if fld in mp_str_fields:
                    f.write(f"        .{fld} = \"{val}\",\n")
                else:
                    f.write(f"        .{fld} = {val},\n")
            f.write("    },\n")
        f.write("};\n\n")

        # Maxpool accessors
        f.write("""\
static inline int get_num_maxpool_layers(void) { return NUM_MAXPOOL_LAYERS; }

static inline const maxpool_layer_config_t* get_maxpool_config(int layer_id) {
    if (layer_id < 0 || layer_id >= NUM_MAXPOOL_LAYERS) return NULL;
    return &MAXPOOL_LAYER_CONFIGS[layer_id];
}

static inline const maxpool_layer_config_t* get_maxpool_config_by_params(
    int channels, int src_height, int src_width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    for (int i = 0; i < NUM_MAXPOOL_LAYERS; i++) {
        const maxpool_layer_config_t* c = &MAXPOOL_LAYER_CONFIGS[i];
        if (c->channels   == channels   &&
            c->src_height == src_height &&
            c->src_width  == src_width  &&
            c->kernel_h   == kernel_h   &&
            c->kernel_w   == kernel_w   &&
            c->stride_h   == stride_h   &&
            c->stride_w   == stride_w   &&
            c->pad_h      == pad_h      &&
            c->pad_w      == pad_w)
            return c;
    }
    return NULL;
}

#endif /* LAYER_CONFIGS_H */
""")

    print(f"\nGenerated {output_file}")
    print(f"  Conv layers:    {len(conv_configs)}")
    print(f"  Maxpool layers: {len(maxpool_configs)}")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate combined conv2d + maxpool DMA config header from PTE files')
    parser.add_argument('--pte', nargs='+', required=True,
                        help='One or more ExecuTorch .pte files')
    parser.add_argument('--output', '-o', default='layer_configs.h',
                        help='Output C header file (default: layer_configs.h)')
    parser.add_argument('--dram0', type=int, default=62976,
                        help='DRAM0 size in bytes (default: 62976)')
    parser.add_argument('--dram1', type=int, default=62976,
                        help='DRAM1 size in bytes (default: 62976)')
    parser.add_argument('--flatc', default=None,
                        help='Path to flatc binary (auto-detected)')
    parser.add_argument('--no-dma-mode', action='store_true', default=False,
                        help='Force all conv kernels to no-DMA mode')

    args = parser.parse_args()

    # Collect layers from all PTE files with deduplication
    all_conv = []
    all_maxpool = []
    conv_seen = set()
    mp_seen = set()

    for pte_path_str in args.pte:
        pte_path = Path(pte_path_str)
        if not pte_path.exists():
            print(f"ERROR: PTE file not found: {pte_path}")
            return 1

        print(f"\nExtracting from: {pte_path}")
        conv_layers, mp_layers = extract_layers_from_pte(pte_path, flatc_path=args.flatc)

        for l in conv_layers:
            key = (l['input'], l['output'], l['kernel'],
                   l['stride'], l['padding'], l['dilation'])
            if key not in conv_seen:
                conv_seen.add(key)
                l['layer_id'] = len(all_conv)
                all_conv.append(l)
            else:
                print(f"  [skip dup conv] {l['name']}")

        for l in mp_layers:
            key = (l['channels'], l['src_height'], l['src_width'],
                   l['kernel_h'], l['kernel_w'], l['stride_h'], l['stride_w'],
                   l['pad_h'], l['pad_w'])
            if key not in mp_seen:
                mp_seen.add(key)
                all_maxpool.append(l)
            else:
                print(f"  [skip dup maxpool] {l['name']}")

    print(f"\nTotal unique: {len(all_conv)} conv, {len(all_maxpool)} maxpool")
    print(f"DRAM budget: DRAM0={args.dram0}B  DRAM1={args.dram1}B")

    # Calculate conv configs
    print(f"\nCalculating conv configurations...")
    conv_configs = []
    for layer in all_conv:
        print(f"  Conv {layer['layer_id']}: {layer['name']}...")
        cfg = calculate_conv_config(layer, args.dram0, args.dram1)
        if cfg:
            conv_configs.append(cfg)
            print(f"    [OK] n_tile={cfg['n_tile_size']}, height_tiles={cfg['height_tiles']}, "
                  f"output_rows={cfg['output_rows']}")
        else:
            print(f"    [FAIL]")

    # Apply no-DMA mode
    if args.no_dma_mode:
        for cfg in conv_configs:
            if cfg['kernel_name'].endswith('_dma'):
                cfg['kernel_name'] = cfg['kernel_name'][:-4] + '_no_dma'
        print("No-DMA mode: all conv kernels set to _no_dma")

    # Calculate maxpool configs
    print(f"\nCalculating maxpool configurations...")
    mp_configs = []
    for idx, layer in enumerate(all_maxpool):
        print(f"  Maxpool {idx}: {layer['name']}...")
        cfg = build_maxpool_config(idx, layer, args.dram0, args.dram1)
        mp_configs.append(cfg)
        print(f"    [OK] c_tile={cfg['c_tile_size']}, height_tiles={cfg['height_tiles']}, "
              f"output_rows={cfg['output_rows']}")

    # Generate combined header
    generate_combined_header(conv_configs, mp_configs, args.output,
                             args.dram0, args.dram1, args.no_dma_mode)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
