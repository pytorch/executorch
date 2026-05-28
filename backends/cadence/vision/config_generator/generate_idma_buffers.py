#!/usr/bin/env python3
"""
Generate IDMA buffer size definitions for convolution operations.

This script calculates buffer sizes based on:
- Processing all width elements in one go
- Processing 2 output rows in one go
- Processing all output channels in one go
"""

# DRAM Size Configuration (in bytes)
DRAM_SIZE_0 = 32 * 1024  # 128 KB for DRAM0
DRAM_SIZE_1 = 32 * 1024  # 64 KB for DRAM1
def find_max_tile_config(input_whd, output_whd, kernel_whdn, padding, stride_xy, kernel_name="7x7j2d1", data_type="S8S8", dram0_size=None, dram1_size=None, conv_params=None, conv_flags=0):
    """
    Find maximum output channels and output rows that fit in available DRAM.
    
    Strategy:
    1. Start with n_tile_size=1, output_rows=1
    2. Increase n_tile_size until all output channels are covered or memory is full
    3. Once all channels fit, increase output_rows
    
    Args:
        input_whd: Tuple (width, height, depth) of input
        output_whd: Tuple (width, height, depth) of output
        kernel_whdn: Tuple (width, height, depth, num_filters) of kernel
        padding: Tuple (dim1_edge1, dim1_edge2, dim2_edge1, dim2_edge2, dim3_edge1, dim3_edge2)
        stride_xy: Tuple (stride_x, stride_y)
        kernel_name: String identifier for kernel
        data_type: Data type string
        dram0_size: Size of DRAM0 in bytes
        dram1_size: Size of DRAM1 in bytes
        conv_params: Tuple of (strideX, strideY, accumShift, reluMax, outputShift, outputScale, dilation, kernelHeight, kernelWidth)
        conv_flags: Integer flags value (e.g., CNN_CONV_FLAG_RELU)
    
    Returns:
        Tuple (best_n_tile_size, best_output_rows, buffer_sizes_dict)
    """
    if dram0_size is None:
        dram0_size = DRAM_SIZE_0
    if dram1_size is None:
        dram1_size = DRAM_SIZE_1
    
    output_w, output_h, output_d = output_whd
    kernel_w, kernel_h, kernel_d, kernel_n = kernel_whdn
    stride_x, stride_y = stride_xy
    
    print(f"\n=== Finding Maximum Tile Configuration ===")
    print(f"Kernel: {kernel_name}")
    print(f"Total output channels: {kernel_n}")
    print(f"DRAM0: {dram0_size} bytes, DRAM1: {dram1_size} bytes")
    print()
    
    best_n_tile_size = 1
    best_output_rows = 2  # Minimum output rows should be 2
    best_buffer_sizes = None
    all_channels_fit = False
    best_tile_balance = float('inf')  # Track tile size balance (lower is better)
    
    # Phase 1: Scan ALL n_tile_size values to find best balanced config
    current_output_rows = 2
    last_fit_n_tile_size = 0
    
    for n_tile_size in range(1, kernel_n + 1):
        # Temporarily modify output_whd for this iteration
        temp_output_whd = (output_w, output_h, output_d)
        
        # Calculate buffer sizes with current configuration
        buffer_sizes = calculate_buffer_sizes_with_rows(
            input_whd, temp_output_whd, kernel_whdn, padding, stride_xy,
            kernel_name, data_type, n_tile_size, current_output_rows,
            conv_params, conv_flags
        )
        
        # Check if it fits in DRAM
        placement = calculate_buffer_placement(buffer_sizes, dram0_size, dram1_size)
        
        if placement['total_fits']:
            last_fit_n_tile_size = n_tile_size
            
            # Calculate tile balance: difference between first tile and last tile
            n_tiles = (kernel_n + n_tile_size - 1) // n_tile_size
            last_tile_size = kernel_n - (n_tile_size * (n_tiles - 1))
            tile_balance = abs(n_tile_size - last_tile_size)
            
            # Update best if this is more balanced than current best
            # OR if balance is same but tile size is larger (prefer fewer, larger tiles)
            if tile_balance < best_tile_balance or \
               (tile_balance == best_tile_balance and n_tile_size > best_n_tile_size):
                best_n_tile_size = n_tile_size
                best_output_rows = current_output_rows
                best_buffer_sizes = buffer_sizes
                best_tile_balance = tile_balance
                
                if n_tile_size >= kernel_n:
                    all_channels_fit = True
                    print(f"  All {kernel_n} output channels fit with {current_output_rows} output rows")
                    break
        else:
            # Stop scanning if we've found at least one config and current doesn't fit
            if last_fit_n_tile_size > 0:
                print(f"  n_tile_size={n_tile_size}, output_rows={current_output_rows}: Does NOT fit")
                break
    
    # Phase 2: If all channels fit, try increasing output_rows
    if all_channels_fit:
        print(f"\n  Phase 2: Increasing output rows (all channels fit)...")
        
        for output_rows in range(3, output_h + 1):
            buffer_sizes = calculate_buffer_sizes_with_rows(
                input_whd, temp_output_whd, kernel_whdn, padding, stride_xy,
                kernel_name, data_type, best_n_tile_size, output_rows,
                conv_params, conv_flags
            )
            
            placement = calculate_buffer_placement(buffer_sizes, dram0_size, dram1_size)
            
            if placement['total_fits']:
                best_output_rows = output_rows
                best_buffer_sizes = buffer_sizes
                print(f"  output_rows={output_rows}: Fits!")
            else:
                print(f"  output_rows={output_rows}: Does NOT fit")
                break
    
    print(f"\n=== Best Configuration Found ===")
    if best_buffer_sizes is None:
        print(f"\033[91m  ERROR: No configuration fits in available DRAM!\033[0m")
        print(f"\033[91m  DRAM0: {dram0_size} bytes, DRAM1: {dram1_size} bytes\033[0m")
        print(f"\033[91m  Minimum required: n_tile_size=1, output_rows=1\033[0m")
        print(f"\033[91m  Setting all buffer sizes to 0 for kernel {kernel_name}\033[0m")
        
        # Create a minimal buffer_sizes dict with all zeros
        best_buffer_sizes = {
            'IN': 0, 'COEFF': 0, 'COEFF_TILE_SIZE_LAST': 0, 'OUT': 0, 'BIAS': 0, 'OUTSCALE': 0,
            'padding': padding, 'kernel_name': kernel_name, 'data_type': data_type,
            'SRC_DIM1_SIZE': 0, 'SRC_DIM1_PITCH': 0, 'SRC_DIM2_SIZE': 0, 'SRC_DIM2_PITCH': 0, 'SRC_DIM3_SIZE': 0,
            'DST_DIM1_SIZE': 0, 'DST_DIM1_PITCH': 0, 'DST_DIM2_SIZE': 0, 'DST_DIM2_PITCH': 0,
            'IN_DIM1_SIZE': 0, 'IN_DIM1_PITCH': 0, 'IN_DIM2_SIZE': 0, 'IN_DIM2_PITCH': 0,
            'IN_DATA_OFFSET': 0, 'IN_ROWS_FIRSTDMA': 0,
            'OUT_DIM1_SIZE': 0, 'OUT_DIM1_PITCH': 0, 'OUT_DIM2_SIZE': 0, 'OUT_DIM2_PITCH': 0, 'OUT_DIM3_SIZE': 0,
            'COEFF_DIM1_SIZE': 0, 'COEFF_DIM2_SIZE': 0, 'COEFF_DIM3_SIZE': 0, 'COEFF_DIM4_SIZE': 0,
            'COEFF_DIM1_PITCH': 0, 'COEFF_DIM2_PITCH': 0, 'COEFF_DIM3_PITCH': 0,
            'BIAS_DIM1_SIZE': 0, 'BIAS_DIM2_SIZE': 0,
            'OUTSCALE_DIM1_SIZE': 0, 'OUTSCALE_DIM2_SIZE': 0,
            'N_TILE_SIZE': 0, 'N_TILES': 0, 'N_TILE_SIZE_LAST': 0, 'HIGHT_TILES': 0,
            'details': {'input_buff_whd': (0, 0, 0), 'input_rows_needed': 0, 'output_buff_whd': (0, 0, 0)}
        }
        best_n_tile_size = 0
        best_output_rows = 0
    
    print(f"  n_tile_size: {best_n_tile_size} (out of {kernel_n} total channels)")
    print(f"  output_rows: {best_output_rows}")
    print()
    
    return best_n_tile_size, best_output_rows, best_buffer_sizes

def calculate_buffer_sizes_with_rows(input_whd, output_whd, kernel_whdn, padding, stride_xy, kernel_name="7x7j2d1", data_type="S8S8", n_tile_size=None, output_rows_per_iteration=2, conv_params=None, conv_flags=0):
    """
    Calculate IDMA buffer sizes with configurable output rows per iteration.
    
    Args:
        Same as calculate_buffer_sizes, plus:
        output_rows_per_iteration: Number of output rows to process in one iteration
        conv_params: Tuple of (strideX, strideY, accumShift, reluMax, outputShift, outputScale, dilation, kernelHeight, kernelWidth)
        conv_flags: Integer flags value (e.g., CNN_CONV_FLAG_RELU)
    
    Returns:
        Dictionary with buffer sizes
    """
    input_w, input_h, input_d = input_whd
    output_w, output_h, output_d = output_whd
    kernel_w, kernel_h, kernel_d, kernel_n = kernel_whdn
    dim1_edge1, dim1_edge2, dim2_edge1, dim2_edge2, dim3_edge1, dim3_edge2 = padding
    stride_x, stride_y = stride_xy
    
    # Calculate input tile dimensions
    # For width (DIM1): we always process full input width (no horizontal tiling)
    input_dim1_size = input_w  # Full width, no horizontal tiling
    
    # Calculate input buffer size
    # For N output rows, we need enough input rows to cover them with the kernel
    input_rows_needed = (output_rows_per_iteration - 1) * stride_y + kernel_h
    
    # Input buffer dimensions (WHD format)
    input_buff_w = input_dim1_size + dim1_edge1 + dim1_edge2
    input_buff_h = input_rows_needed
    input_buff_d = input_d + dim3_edge1 + dim3_edge2
    
    # Input buffer size in bytes
    input_buff_size = input_buff_w * input_buff_h * input_buff_d
    
    # Tiling parameters
    if n_tile_size is None:
        n_tile_size_val = kernel_n
        n_tiles = 1
        n_tile_size_last = kernel_n
    else:
        n_tile_size_val = n_tile_size
        n_tiles = (kernel_n + n_tile_size - 1) // n_tile_size
        n_tile_size_last = kernel_n - (n_tile_size * (n_tiles - 1))
    
    # Coefficient buffer size
    coeff_buff_size = kernel_w * kernel_h * kernel_d * n_tile_size_val
    
    # Coefficient tile size for last tile
    coeff_tile_size_last = kernel_w * kernel_h * kernel_d * n_tile_size_last
    
    # Calculate output buffer size
    output_buff_w = output_w
    output_buff_h = output_rows_per_iteration
    output_buff_d = n_tile_size_val
    output_buff_size = output_buff_w * output_buff_h * output_buff_d
    
    # Bias and outscale buffers
    bias_buff_size = kernel_n * 4  # S32
    outscale_buff_size = kernel_n * 2  # U16
    
    # Calculate tile dimensions and pitches
# Calculate tile dimensions and pitches
    src_dim1_size = input_w
    src_dim1_pitch = input_w
    src_dim2_size = input_h
    src_dim2_pitch = input_w * input_h
    src_dim3_size = input_d
    
    dst_dim1_size = output_w
    dst_dim1_pitch = output_w
    dst_dim2_size = output_h
    dst_dim2_pitch = output_w * output_h
    
    in_dim1_size = input_dim1_size
    in_dim1_pitch = input_buff_w  # DIM1_PITCH = row size + dim1 padding
    in_dim2_size = input_rows_needed
    in_dim2_pitch = in_dim2_size * in_dim1_pitch  # DIM2_PITCH = DIM2_SIZE * DIM1_PITCH
    
    in_data_offset = (dim2_edge1 * in_dim1_pitch) + dim1_edge1
    in_rows_firstdma = input_rows_needed - dim2_edge1
    
    out_dim1_size = output_w
    out_dim1_pitch = output_w
    out_dim2_size = output_rows_per_iteration
    out_dim2_pitch = output_w * output_rows_per_iteration
    out_dim3_size = n_tile_size_val
    
    coeff_dim1_size = kernel_w
    coeff_dim2_size = kernel_h
    coeff_dim3_size = kernel_d
    coeff_dim4_size = kernel_n
    coeff_dim1_pitch = kernel_w
    coeff_dim2_pitch = kernel_w * kernel_h
    coeff_dim3_pitch = kernel_w * kernel_h * kernel_d
    
    bias_dim1_size = kernel_n
    bias_dim2_size = 1
    
    outscale_dim1_size = kernel_n
    outscale_dim2_size = 1
    
    height_tiles = (output_h + output_rows_per_iteration - 1) // output_rows_per_iteration
    
    result = {
        'IN': input_buff_size,
        'COEFF': coeff_buff_size,
        'COEFF_TILE_SIZE_LAST': coeff_tile_size_last,
        'OUT': output_buff_size,
        'BIAS': bias_buff_size,
        'OUTSCALE': outscale_buff_size,
        'padding': padding,
        'kernel_name': kernel_name,
        'data_type': data_type,
        'SRC_DIM1_SIZE': src_dim1_size,
        'SRC_DIM1_PITCH': src_dim1_pitch,
        'SRC_DIM2_SIZE': src_dim2_size,
        'SRC_DIM2_PITCH': src_dim2_pitch,
        'SRC_DIM3_SIZE': src_dim3_size,
        'DST_DIM1_SIZE': dst_dim1_size,
        'DST_DIM1_PITCH': dst_dim1_pitch,
        'DST_DIM2_SIZE': dst_dim2_size,
        'DST_DIM2_PITCH': dst_dim2_pitch,
        'IN_DIM1_SIZE': in_dim1_size,
        'IN_DIM1_PITCH': in_dim1_pitch,
        'IN_DIM2_SIZE': in_dim2_size,
        'IN_DIM2_PITCH': in_dim2_pitch,
        'IN_DATA_OFFSET': in_data_offset,
        'IN_ROWS_FIRSTDMA': in_rows_firstdma,
        'OUT_DIM1_SIZE': out_dim1_size,
        'OUT_DIM1_PITCH': out_dim1_pitch,
        'OUT_DIM2_SIZE': out_dim2_size,
        'OUT_DIM2_PITCH': out_dim2_pitch,
        'OUT_DIM3_SIZE': out_dim3_size,
        'COEFF_DIM1_SIZE': coeff_dim1_size,
        'COEFF_DIM2_SIZE': coeff_dim2_size,
        'COEFF_DIM3_SIZE': coeff_dim3_size,
        'COEFF_DIM4_SIZE': coeff_dim4_size,
        'COEFF_DIM1_PITCH': coeff_dim1_pitch,
        'COEFF_DIM2_PITCH': coeff_dim2_pitch,
        'COEFF_DIM3_PITCH': coeff_dim3_pitch,
        'BIAS_DIM1_SIZE': bias_dim1_size,
        'BIAS_DIM2_SIZE': bias_dim2_size,
        'OUTSCALE_DIM1_SIZE': outscale_dim1_size,
        'OUTSCALE_DIM2_SIZE': outscale_dim2_size,
        'N_TILE_SIZE': n_tile_size_val,
        'N_TILES': n_tiles,
        'N_TILE_SIZE_LAST': n_tile_size_last,
        'HIGHT_TILES': height_tiles,
        'details': {
            'input_buff_whd': (input_buff_w, input_buff_h, input_buff_d),
            'input_rows_needed': input_rows_needed,
            'output_buff_whd': (output_buff_w, output_buff_h, output_buff_d),
        }
    }
    
    # Add convolution parameters if provided
    if conv_params is not None:
        strideX, strideY, accumShift, reluMax, outputShift, outputScale, dilation, kernelHeight, kernelWidth = conv_params
        result.update({
            'STRIDEX': strideX,
            'STRIDEY': strideY,
            'ACCUM_SHIFT': accumShift,
            'RELU_MAX': reluMax,
            'RELU_MIN': 0,  # Default minimum
            'OUTPUT_SHIFT': outputShift,
            'OUTPUT_SCALE': outputScale,
            'DILATION': dilation,
            'KERNEL_HEIGHT': kernelHeight,
            'KERNEL_WIDTH': kernelWidth,
            'FLAGS': conv_flags,
        })
    
    return result

def calculate_buffer_sizes(input_whd, output_whd, kernel_whdn, padding, stride_xy, kernel_name="7x7j2d1", data_type="S8S8", n_tile_size=None):
    """
    Calculate IDMA buffer sizes for convolution (uses default 2 output rows).
    
    This is a wrapper around calculate_buffer_sizes_with_rows with output_rows=2.
    """
    return calculate_buffer_sizes_with_rows(input_whd, output_whd, kernel_whdn, padding, stride_xy, kernel_name, data_type, n_tile_size, output_rows_per_iteration=2)
    """
    Calculate IDMA buffer sizes for convolution.
    
    Args:
        input_whd: Tuple (width, height, depth) of input
        output_whd: Tuple (width, height, depth) of output
        kernel_whdn: Tuple (width, height, depth, num_filters) of kernel
        padding: Tuple (dim1_edge1, dim1_edge2, dim2_edge1, dim2_edge2, dim3_edge1, dim3_edge2)
        stride_xy: Tuple (stride_x, stride_y)
        kernel_name: String identifier for kernel (e.g., "7x7j2d1")
        data_type: Data type string (e.g., "S8S8")
        n_tile_size: Number of output channels per tile (None = all channels)
    
    Returns:
        Dictionary with buffer sizes
    """
    input_w, input_h, input_d = input_whd
    output_w, output_h, output_d = output_whd
    kernel_w, kernel_h, kernel_d, kernel_n = kernel_whdn
    dim1_edge1, dim1_edge2, dim2_edge1, dim2_edge2, dim3_edge1, dim3_edge2 = padding
    stride_x, stride_y = stride_xy
    
    # Assumptions:
    # - Process all width elements in one go
    # - Process 2 output rows in one go
    # - Process all output channels in one go
    
    output_rows_per_iteration = 2
    
    # Calculate input tile dimensions
    # DIM1_SIZE for input processing
    input_dim1_size = input_w - stride_x + 1
    
    # Calculate input buffer size
    # For 2 output rows, we need enough input rows to cover them with the kernel
    input_rows_needed = (output_rows_per_iteration - 1) * stride_y + kernel_h
    
    # Input buffer dimensions (WHD format)
    # Width includes padding
    input_buff_w = input_dim1_size + dim1_edge1 + dim1_edge2
    # Height is just the rows needed (padding is NOT added to height for buffer calculation)
    input_buff_h = input_rows_needed
    # Depth includes padding
    input_buff_d = input_d + dim3_edge1 + dim3_edge2
    
    # Input buffer size in bytes (assuming 1 byte per element for S8)
    input_buff_size = input_buff_w * input_buff_h * input_buff_d
    
    # Tiling parameters (calculate early to use in buffer sizes)
    if n_tile_size is None:
        # No tiling - process all output channels at once
        n_tile_size_val = kernel_n
        n_tiles = 1
        n_tile_size_last = kernel_n
    else:
        # Calculate number of tiles needed
        n_tile_size_val = n_tile_size
        n_tiles = (kernel_n + n_tile_size - 1) // n_tile_size  # Ceiling division
        n_tile_size_last = kernel_n - (n_tile_size * (n_tiles - 1))
    
    # Coefficient buffer size (only one tile worth of coefficients at a time)
    coeff_buff_size = kernel_w * kernel_h * kernel_d * n_tile_size_val
    
    # Calculate coefficient tile size for last tile
    coeff_tile_size_last = kernel_w * kernel_h * kernel_d * n_tile_size_last
    
    # Calculate output buffer size (WHD format)
    # Process all width, 2 rows, n_tile_size channels
    output_buff_w = output_w
    output_buff_h = output_rows_per_iteration
    output_buff_d = n_tile_size_val  # Use tile size instead of total output channels
    output_buff_size = output_buff_w * output_buff_h * output_buff_d
    
    # Calculate bias buffer size (S32 = 4 bytes per element)
    # One bias value per output channel (for all channels, not just one tile)
    bias_buff_size = kernel_n * 4  # S32 uses 4 bytes
    
    # Calculate output scale buffer size (U16 = 2 bytes per element)
    # One scale value per output channel (for all channels, not just one tile)
    outscale_buff_size = kernel_n * 2  # U16 uses 2 bytes
    
    # Calculate tile dimensions and pitches
    # Source tile parameters (original input dimensions)
    src_dim1_size = input_w
    src_dim1_pitch = input_w
    src_dim2_size = input_h
    src_dim2_pitch = input_w * input_h
    src_dim3_size = input_d
    
    # Destination tile parameters (original output dimensions)
    dst_dim1_size = output_w
    dst_dim1_pitch = output_w
    dst_dim2_size = output_h
    dst_dim2_pitch = output_w * output_h
    
    # Input tile (WHD format with padding)
    input_dim1_size = input_w - stride_x + 1  # Width for single tile processing
    input_dim1_pitch = input_dim1_size + dim1_edge1 + dim1_edge2  # Width with padding
    input_dim2_size = input_rows_needed  # Number of input rows (kernel_h + (output_rows-1)*stride_y)
    input_dim2_pitch = input_dim1_pitch * input_rows_needed  # Pitch for next depth plane (rows × width)
    
    # Calculate data offset (padding offset in the buffer)
    # Offset = (top_padding_rows * pitch) + left_padding_pixels
    input_data_offset = (dim2_edge1 * input_dim1_pitch) + dim1_edge1
    
    # Calculate rows for first DMA (excludes top padding)
    input_rows_firstdma = input_rows_needed - dim2_edge1
    
    # Output tile (WHD format)
    output_dim1_size = output_buff_w
    output_dim1_pitch = output_buff_w
    output_dim2_size = output_rows_per_iteration
    output_dim2_pitch = output_rows_per_iteration * output_dim1_pitch  # Output rows in one go × width
    output_dim3_size = n_tile_size_val  # Use tile size
    
    # Coefficient tile parameters (WHDN format)
    coeff_dim1_size = kernel_w
    coeff_dim2_size = kernel_h
    coeff_dim3_size = kernel_d
    coeff_dim4_size = kernel_n
    coeff_dim1_pitch = kernel_w
    coeff_dim2_pitch = kernel_w * kernel_h
    coeff_dim3_pitch = kernel_w * kernel_h * kernel_d
    
    # Bias array parameters
    bias_dim1_size = kernel_n
    bias_dim2_size = 1
    
    # Output scale array parameters
    outscale_dim1_size = kernel_n
    outscale_dim2_size = 1
    
    # Height tiles (number of iterations for output height)
    height_tiles = output_h // output_rows_per_iteration
    
    return {
        'IN': input_buff_size,
        'COEFF': coeff_buff_size,
        'COEFF_TILE_SIZE_LAST': coeff_tile_size_last,
        'OUT': output_buff_size,
        'BIAS': bias_buff_size,
        'OUTSCALE': outscale_buff_size,
        'kernel_name': kernel_name,
        'data_type': data_type,
        'padding': padding,
        'SRC_DIM1_SIZE': src_dim1_size,
        'SRC_DIM1_PITCH': src_dim1_pitch,
        'SRC_DIM2_SIZE': src_dim2_size,
        'SRC_DIM2_PITCH': src_dim2_pitch,
        'SRC_DIM3_SIZE': src_dim3_size,
        'DST_DIM1_SIZE': dst_dim1_size,
        'DST_DIM1_PITCH': dst_dim1_pitch,
        'DST_DIM2_SIZE': dst_dim2_size,
        'DST_DIM2_PITCH': dst_dim2_pitch,
        'DIM1_SIZE': input_dim1_size,
        'DIM1_PITCH': input_dim1_pitch,
        'DIM2_SIZE': input_dim2_size,
        'DIM2_PITCH': input_dim2_pitch,
        'IN_DATA_OFFSET': input_data_offset,
        'IN_ROWS_FIRSTDMA': input_rows_firstdma,
        'OUT_DIM1_SIZE': output_dim1_size,
        'OUT_DIM1_PITCH': output_dim1_pitch,
        'OUT_DIM2_SIZE': output_dim2_size,
        'OUT_DIM2_PITCH': output_dim2_pitch,
        'OUT_DIM3_SIZE': output_dim3_size,
        'COEFF_DIM1_SIZE': coeff_dim1_size,
        'COEFF_DIM2_SIZE': coeff_dim2_size,
        'COEFF_DIM3_SIZE': coeff_dim3_size,
        'COEFF_DIM4_SIZE': coeff_dim4_size,
        'COEFF_DIM1_PITCH': coeff_dim1_pitch,
        'COEFF_DIM2_PITCH': coeff_dim2_pitch,
        'COEFF_DIM3_PITCH': coeff_dim3_pitch,
        'BIAS_DIM1_SIZE': bias_dim1_size,
        'BIAS_DIM2_SIZE': bias_dim2_size,
        'OUTSCALE_DIM1_SIZE': outscale_dim1_size,
        'OUTSCALE_DIM2_SIZE': outscale_dim2_size,
        'N_TILE_SIZE': n_tile_size_val,
        'N_TILES': n_tiles,
        'N_TILE_SIZE_LAST': n_tile_size_last,
        'HIGHT_TILES': height_tiles,
        'details': {
            'input_buff_whd': (input_buff_w, input_buff_h, input_buff_d),
            'input_rows_needed': input_rows_needed,
            'output_buff_whd': (output_buff_w, output_buff_h, output_buff_d),
        }
    }


def generate_header_content(buffer_sizes, header_guard="CONVIDMA_BUFFERS_H_"):
    """
    Generate C header file content with buffer size definitions.
    
    Args:
        buffer_sizes: Dictionary from calculate_buffer_sizes()
        header_guard: Header guard name
    
    Returns:
        String containing header file content
    """
    kernel_name = buffer_sizes['kernel_name']
    data_type = buffer_sizes['data_type']
    
    header = f"""/*
 * convIdma_buffers.h
 *
 *  Auto-generated buffer size definitions
 */

#ifndef {header_guard}
#define {header_guard}

// ============================================================================
// IDMA Buffer Sizes and Tile Parameters for convVQ3D_{kernel_name}_{data_type}_MOW_WHD
// ============================================================================

// SRC tile parameters
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_SRC_DIM1_SIZE     {buffer_sizes['SRC_DIM1_SIZE']} // input width
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_SRC_DIM1_PITCH    {buffer_sizes['SRC_DIM1_PITCH']} //
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_SRC_DIM2_SIZE     {buffer_sizes['SRC_DIM2_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_SRC_DIM2_PITCH    {buffer_sizes['SRC_DIM2_PITCH']} // {buffer_sizes['SRC_DIM1_SIZE']}*{buffer_sizes['SRC_DIM2_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_SRC_DIM3_SIZE     {buffer_sizes['SRC_DIM3_SIZE']}

// DST   tile parameters
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_DST_DIM1_SIZE     {buffer_sizes['DST_DIM1_SIZE']} // input width
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_DST_DIM1_PITCH    {buffer_sizes['DST_DIM1_PITCH']} //
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_DST_DIM2_SIZE     {buffer_sizes['DST_DIM2_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_DST_DIM2_PITCH    {buffer_sizes['DST_DIM2_PITCH']} // {buffer_sizes['DST_DIM1_SIZE']}*{buffer_sizes['DST_DIM2_SIZE']}


// Input tile parameters
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_IN_DIM1_SIZE     {buffer_sizes['IN_DIM1_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_IN_DIM1_PITCH    {buffer_sizes['IN_DIM1_PITCH']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_IN_DIM2_SIZE     {buffer_sizes['IN_DIM2_SIZE']} // IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM2_SIZE + ((IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM2_SIZE-1)* stride)
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_IN_DIM2_PITCH    {buffer_sizes['IN_DIM2_PITCH']}


#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_IN_DATA_OFFSET   {buffer_sizes['IN_DATA_OFFSET']}
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_IN_ROWS_FIRSTDMA {buffer_sizes['IN_ROWS_FIRSTDMA']}  // IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_IN_DIM2_SIZE - padding rows 
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_IN_FRAME_PTR 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_IN_STATUS_FLAGS 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_IN_DIM1_COORD 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_IN_DIM3_COORD 0

// Output tile parameters
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM1_SIZE     {buffer_sizes['OUT_DIM1_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM1_PITCH    {buffer_sizes['OUT_DIM1_PITCH']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM2_SIZE     {buffer_sizes['OUT_DIM2_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM2_PITCH    {buffer_sizes['OUT_DIM2_PITCH']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM3_SIZE     {buffer_sizes['N_TILE_SIZE']} //IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_N_TILE_SIZE
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT_FRAME_PTR 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT_STATUS_FLAGS 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM1_COORD 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM1_EDGE1 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM1_EDGE2 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM2_EDGE1 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM2_EDGE2 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM3_COORD 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM3_EDGE1 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM3_EDGE2 0

//coefficient tile parameters
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM1_SIZE     {buffer_sizes['COEFF_DIM1_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM2_SIZE     {buffer_sizes['COEFF_DIM2_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM3_SIZE     {buffer_sizes['COEFF_DIM3_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM4_SIZE     {buffer_sizes['COEFF_DIM4_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM1_PITCH    {buffer_sizes['COEFF_DIM1_PITCH']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM2_PITCH    {buffer_sizes['COEFF_DIM2_PITCH']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM3_PITCH    {buffer_sizes['COEFF_DIM3_PITCH']}

#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_FRAME_PTR 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_STATUS_FLAGS 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM1_COORD 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM1_EDGE1 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM1_EDGE2 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM2_COORD 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM2_EDGE1 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM2_EDGE2 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM3_COORD 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM3_EDGE1 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM3_EDGE2 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM4_COORD 0

//bias array parameters
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_BIAS_DIM1_SIZE       {buffer_sizes['BIAS_DIM1_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_BIAS_DIM2_SIZE       {buffer_sizes['BIAS_DIM2_SIZE']}       

//output scale array parameters
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_OUTSCALE_DIM1_SIZE     {buffer_sizes['OUTSCALE_DIM1_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_OUTSCALE_DIM2_SIZE     {buffer_sizes['OUTSCALE_DIM2_SIZE']}

// Buffer sizes
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_IN       {buffer_sizes['IN']}
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF    {buffer_sizes['COEFF']}
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT      {buffer_sizes['OUT']}
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_BIAS     {buffer_sizes['BIAS']}
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUTSCALE {buffer_sizes['OUTSCALE']}

#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_N_TILES {buffer_sizes['N_TILES']}           // round_toward positive(IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_SRC_DIM3_SIZE / IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_N_TILE_SIZE)
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_HIGHT_TILES {buffer_sizes['HIGHT_TILES']}     //IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_DST_DIM2_SIZE / IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM2_SIZE   
#define  IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_N_TILE_SIZE {buffer_sizes['N_TILE_SIZE']}    // take this as input aas of now (contstant 22 for 3x3 conv and constant 64 for 7x7 conv)
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_N_TILE_SIZE_LAST {buffer_sizes['N_TILE_SIZE_LAST']} //IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM4_SIZE - IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_N_TILE_SIZE

#endif /* {header_guard} */
"""
    return header


def align_to_64(size):
    """Round up size to next 64-byte boundary for alignment."""
    return ((size + 63) // 64) * 64


def calculate_buffer_placement(buffer_sizes, dram0_size=32*1024, dram1_size=32*1024):
    """
    Calculate optimal buffer placement in DRAM0 and DRAM1 for ping-pong architecture.
    
    Strategy:
    1. Try default placement: input/coeff in DRAM0, output/bias/outscale in DRAM1
    2. If DRAM0 overflows, move coefficient to DRAM1 (if it fits)
    3. If DRAM1 overflows, move bias/outscale to DRAM0 (if it fits)
    4. Report best fit or overflow scenario
    
    Note: All buffer sizes are aligned to 64 bytes to account for alignment overhead.
    
    Args:
        buffer_sizes: Dictionary from calculate_buffer_sizes()
        dram0_size: Size of DRAM0 in bytes (default 32KB)
        dram1_size: Size of DRAM1 in bytes (default: use global DRAM_SIZE_1)
    
    Returns:
        Dictionary with buffer placement information
    """
    # Use global DRAM sizes if not specified
    if dram0_size is None:
        dram0_size = DRAM_SIZE_0
    if dram1_size is None:
        dram1_size = DRAM_SIZE_1
    
    # Ping-pong buffers require 2x allocation
    # Align each buffer to 64 bytes to account for alignment overhead
    input_ping = align_to_64(buffer_sizes['IN'])
    input_pong = align_to_64(buffer_sizes['IN'])
    coeff = align_to_64(buffer_sizes['COEFF'])
    output_ping = align_to_64(buffer_sizes['OUT'])
    output_pong = align_to_64(buffer_sizes['OUT'])
    bias = align_to_64(buffer_sizes['BIAS'])
    outscale = align_to_64(buffer_sizes['OUTSCALE'])
    
    # CRITICAL: Check if any single buffer exceeds DRAM bank size
    # A single buffer cannot be split across banks, so each must fit individually
    max_bank_size = max(dram0_size, dram1_size)
    if coeff > max_bank_size:
        # Coefficient buffer too large - cannot fit in any single DRAM bank
        return {
            'strategy': 'FAIL_COEFF_TOO_LARGE',
            'dram0_allocation': [],
            'dram1_allocation': [],
            'dram0_used': 0,
            'dram1_used': 0,
            'dram0_size': dram0_size,
            'dram1_size': dram1_size,
            'dram0_free': dram0_size,
            'dram1_free': dram1_size,
            'dram0_fits': False,
            'dram1_fits': False,
            'total_fits': False,
            'error': f'Coefficient buffer ({coeff} bytes) exceeds max DRAM bank size ({max_bank_size} bytes)'
        }
    if input_ping > max_bank_size:
        return {
            'strategy': 'FAIL_INPUT_TOO_LARGE',
            'dram0_allocation': [],
            'dram1_allocation': [],
            'dram0_used': 0,
            'dram1_used': 0,
            'dram0_size': dram0_size,
            'dram1_size': dram1_size,
            'dram0_free': dram0_size,
            'dram1_free': dram1_size,
            'dram0_fits': False,
            'dram1_fits': False,
            'total_fits': False,
            'error': f'Input buffer ({input_ping} bytes) exceeds max DRAM bank size ({max_bank_size} bytes)'
        }
    if output_ping > max_bank_size:
        return {
            'strategy': 'FAIL_OUTPUT_TOO_LARGE',
            'dram0_allocation': [],
            'dram1_allocation': [],
            'dram0_used': 0,
            'dram1_used': 0,
            'dram0_size': dram0_size,
            'dram1_size': dram1_size,
            'dram0_free': dram0_size,
            'dram1_free': dram1_size,
            'dram0_fits': False,
            'dram1_fits': False,
            'total_fits': False,
            'error': f'Output buffer ({output_ping} bytes) exceeds max DRAM bank size ({max_bank_size} bytes)'
        }
    
    # Strategy 1: Default placement - input/coeff in DRAM0, output/bias/outscale in DRAM1
    strategy = "default"
    dram0_allocation = [
        ('input_ping', input_ping),
        ('input_pong', input_pong),
        ('coeff', coeff)
    ]
    dram1_allocation = [
        ('output_ping', output_ping),
        ('output_pong', output_pong),
        ('bias', bias),
        ('outscale', outscale)
    ]
    
    dram0_used = sum(size for _, size in dram0_allocation)
    dram1_used = sum(size for _, size in dram1_allocation)
    dram0_fits = dram0_used <= dram0_size
    dram1_fits = dram1_used <= dram1_size
    
    # Strategy 2: If DRAM0 overflows, try moving coefficient to DRAM1
    if not dram0_fits and (dram1_used + coeff <= dram1_size):
        strategy = "coeff_to_dram1"
        dram0_allocation = [
            ('input_ping', input_ping),
            ('input_pong', input_pong)
        ]
        dram1_allocation = [
            ('coeff', coeff),
            ('output_ping', output_ping),
            ('output_pong', output_pong),
            ('bias', bias),
            ('outscale', outscale)
        ]
        dram0_used = sum(size for _, size in dram0_allocation)
        dram1_used = sum(size for _, size in dram1_allocation)
        dram0_fits = dram0_used <= dram0_size
        dram1_fits = dram1_used <= dram1_size
    
    # Strategy 3: If DRAM1 overflows, try moving bias/outscale to DRAM0
    elif not dram1_fits and (dram0_used + bias + outscale <= dram0_size):
        strategy = "bias_outscale_to_dram0"
        dram0_allocation = [
            ('input_ping', input_ping),
            ('input_pong', input_pong),
            ('coeff', coeff),
            ('bias', bias),
            ('outscale', outscale)
        ]
        dram1_allocation = [
            ('output_ping', output_ping),
            ('output_pong', output_pong)
        ]
        dram0_used = sum(size for _, size in dram0_allocation)
        dram1_used = sum(size for _, size in dram1_allocation)
        dram0_fits = dram0_used <= dram0_size
        dram1_fits = dram1_used <= dram1_size
    
    # Strategy 4: Try combined optimization - coeff+bias+outscale to DRAM1
    if not (dram0_fits and dram1_fits):
        temp_dram0 = [('input_ping', input_ping), ('input_pong', input_pong)]
        temp_dram1 = [('coeff', coeff), ('output_ping', output_ping), ('output_pong', output_pong), 
                      ('bias', bias), ('outscale', outscale)]
        temp_dram0_used = sum(size for _, size in temp_dram0)
        temp_dram1_used = sum(size for _, size in temp_dram1)
        
        if temp_dram0_used <= dram0_size and temp_dram1_used <= dram1_size:
            strategy = "input_only_dram0"
            dram0_allocation = temp_dram0
            dram1_allocation = temp_dram1
            dram0_used = temp_dram0_used
            dram1_used = temp_dram1_used
            dram0_fits = True
            dram1_fits = True
    
    # Strategy 5: Split input ping-pong buffers across DRAMs
    if not (dram0_fits and dram1_fits):
        temp_dram0 = [('input_ping', input_ping), ('coeff', coeff)]
        temp_dram1 = [('input_pong', input_pong), ('output_ping', output_ping), ('output_pong', output_pong), 
                      ('bias', bias), ('outscale', outscale)]
        temp_dram0_used = sum(size for _, size in temp_dram0)
        temp_dram1_used = sum(size for _, size in temp_dram1)
        
        if temp_dram0_used <= dram0_size and temp_dram1_used <= dram1_size:
            strategy = "split_input_ping_pong"
            dram0_allocation = temp_dram0
            dram1_allocation = temp_dram1
            dram0_used = temp_dram0_used
            dram1_used = temp_dram1_used
            dram0_fits = True
            dram1_fits = True
    
    # Strategy 6: Alternative split - input_pong+coeff in DRAM0
    if not (dram0_fits and dram1_fits):
        temp_dram0 = [('input_pong', input_pong), ('coeff', coeff)]
        temp_dram1 = [('input_ping', input_ping), ('output_ping', output_ping), ('output_pong', output_pong), 
                      ('bias', bias), ('outscale', outscale)]
        temp_dram0_used = sum(size for _, size in temp_dram0)
        temp_dram1_used = sum(size for _, size in temp_dram1)
        
        if temp_dram0_used <= dram0_size and temp_dram1_used <= dram1_size:
            strategy = "split_input_ping_pong_alt"
            dram0_allocation = temp_dram0
            dram1_allocation = temp_dram1
            dram0_used = temp_dram0_used
            dram1_used = temp_dram1_used
            dram0_fits = True
            dram1_fits = True
    
    # Build individual buffer DRAM placement mapping from allocation lists
    # Check which DRAM each buffer is allocated to
    dram0_buffers = {name for name, _ in dram0_allocation}
    
    # Map buffer names to their DRAM placement (0 or 1)
    # Default to DRAM1 if not in DRAM0
    in1_dram = 0 if 'input_ping' in dram0_buffers else 1
    in2_dram = 0 if 'input_pong' in dram0_buffers else 1
    coeff_dram = 0 if 'coeff' in dram0_buffers else 1
    out1_dram = 0 if 'output_ping' in dram0_buffers else 1
    out2_dram = 0 if 'output_pong' in dram0_buffers else 1
    bias_dram = 0 if 'bias' in dram0_buffers else 1
    outscale_dram = 0 if 'outscale' in dram0_buffers else 1
    
    return {
        'strategy': strategy,
        'dram0_allocation': dram0_allocation,
        'dram1_allocation': dram1_allocation,
        'dram0_used': dram0_used,
        'dram1_used': dram1_used,
        'dram0_size': dram0_size,
        'dram1_size': dram1_size,
        'dram0_free': dram0_size - dram0_used,
        'dram1_free': dram1_size - dram1_used,
        'dram0_fits': dram0_fits,
        'dram1_fits': dram1_fits,
        'total_fits': dram0_fits and dram1_fits,
        # Individual buffer DRAM placement for generate_layer_configs.py
        'IN1_dram': in1_dram,
        'IN2_dram': in2_dram,
        'COEFF_dram': coeff_dram,
        'OUT1_dram': out1_dram,
        'OUT2_dram': out2_dram,
        'BIAS_dram': bias_dram,
        'OUTSCALE_dram': outscale_dram,
    }


def print_buffer_placement(placement):
    """Print buffer placement information."""
    strategy_names = {
        'default': 'Default: Input+Coeff->DRAM0, Output+Bias+Outscale->DRAM1',
        'coeff_to_dram1': 'Optimized: Coefficient moved to DRAM1',
        'bias_outscale_to_dram0': 'Optimized: Bias+Outscale moved to DRAM0',
        'input_only_dram0': 'Optimized: Only Input ping-pong in DRAM0',
        'split_input_ping_pong': 'Optimized: Input buffers split across DRAMs (ping in DRAM0, pong in DRAM1)',
        'split_input_ping_pong_alt': 'Optimized: Input buffers split across DRAMs (pong in DRAM0, ping in DRAM1)'
    }
    
    print("\n=== Buffer Placement ===")
    print(f"Strategy: {strategy_names.get(placement['strategy'], placement['strategy'])}")
    print(f"DRAM0 Size: {placement['dram0_size']:6d} bytes ({placement['dram0_size']//1024}KB)")
    print(f"DRAM1 Size: {placement['dram1_size']:6d} bytes ({placement['dram1_size']//1024}KB)")
    
    print("\nDRAM0 Allocation:")
    for name, size in placement['dram0_allocation']:
        print(f"  {name:20s} -> {size:6d} bytes -> DRAM0")
    print(f"  {'Total Used':20s}    {placement['dram0_used']:6d} bytes")
    print(f"  {'Free':20s}    {placement['dram0_free']:6d} bytes")
    print(f"  Status: {'OK FITS' if placement['dram0_fits'] else 'X OVERFLOW'}")
    
    print("\nDRAM1 Allocation:")
    for name, size in placement['dram1_allocation']:
        print(f"  {name:20s} -> {size:6d} bytes -> DRAM1")
    print(f"  {'Total Used':20s}    {placement['dram1_used']:6d} bytes")
    print(f"  {'Free':20s}    {placement['dram1_free']:6d} bytes")
    print(f"  Status: {'OK FITS' if placement['dram1_fits'] else 'X OVERFLOW'}")
    
    print(f"\nOverall: {'OK ALL BUFFERS FIT' if placement['total_fits'] else 'X INSUFFICIENT MEMORY'}")


def print_buffer_info(buffer_sizes):
    """Print detailed buffer information."""
    print("\n=== Buffer Size Calculations ===")
    print(f"Kernel: {buffer_sizes['kernel_name']}")
    print(f"Data Type: {buffer_sizes['data_type']}")
    print(f"\nBuffer Sizes:")
    print(f"  INPUT:      {buffer_sizes['IN']:6d} bytes")
    print(f"  COEFF:      {buffer_sizes['COEFF']:6d} bytes")
    print(f"  OUTPUT:     {buffer_sizes['OUT']:6d} bytes")
    print(f"  BIAS:       {buffer_sizes['BIAS']:6d} bytes")
    print(f"  OUTSCALE:   {buffer_sizes['OUTSCALE']:6d} bytes")
    print(f"\nTotal Memory: {sum([buffer_sizes['IN'], buffer_sizes['COEFF'], buffer_sizes['OUT'], buffer_sizes['BIAS'], buffer_sizes['OUTSCALE']]):6d} bytes")
    
    details = buffer_sizes['details']
    print(f"\nTile Parameters:")
    print(f"  SRC_DIM1_SIZE:              {buffer_sizes['SRC_DIM1_SIZE']}")
    print(f"  SRC_DIM1_PITCH:             {buffer_sizes['SRC_DIM1_PITCH']}")
    print(f"  SRC_DIM2_PITCH:             {buffer_sizes['SRC_DIM2_PITCH']}")
    print(f"  DST_DIM1_SIZE:              {buffer_sizes['DST_DIM1_SIZE']}")
    print(f"  DST_DIM1_PITCH:             {buffer_sizes['DST_DIM1_PITCH']}")
    print(f"  DST_DIM2_PITCH:             {buffer_sizes['DST_DIM2_PITCH']}")
    print(f"  DIM1_SIZE (input width):    {buffer_sizes['IN_DIM1_SIZE']}")
    print(f"  DIM1_PITCH (with padding):  {buffer_sizes['IN_DIM1_PITCH']}")
    print(f"  DIM2_PITCH:                 {buffer_sizes['IN_DIM2_PITCH']}")
    print(f"  IN_DATA_OFFSET:             {buffer_sizes['IN_DATA_OFFSET']}")
    print(f"  OUT_DIM1_SIZE:              {buffer_sizes['OUT_DIM1_SIZE']}")
    print(f"  OUT_DIM1_PITCH:             {buffer_sizes['OUT_DIM1_PITCH']}")
    print(f"  OUT_DIM2_SIZE:              {buffer_sizes['OUT_DIM2_SIZE']}")
    print(f"  OUT_DIM2_PITCH:             {buffer_sizes['OUT_DIM2_PITCH']}")
    print(f"  OUT_DIM3_SIZE:              {buffer_sizes['OUT_DIM3_SIZE']}")
    print(f"\nDetails:")
    print(f"  Input buffer WHD (with padding): {details['input_buff_whd']}")
    print(f"  Input rows needed for 2 output rows: {details['input_rows_needed']}")
    print(f"  Output buffer WHD: {details['output_buff_whd']}")


def calculate_conv_params(n, c, h, w, oc, wc, wh, ww, oh, ow, 
                         stride_h, stride_w, padding_h, padding_w, 
                         dilation_h, dilation_w, groups, 
                         in_zero_point, weight_zero_point, 
                         bias_scale, output_scale, output_zero_point):
    """
    Calculate convolution parameters including output_shift and output_scale.
    
    Args:
        n, c, h, w: Input batch, channels, height, width
        oc, wc, wh, ww: Output channels, weight channels, weight height, weight width
        oh, ow: Output height, output width
        stride_h, stride_w: Stride values
        padding_h, padding_w: Padding values
        dilation_h, dilation_w: Dilation values
        groups: Number of groups for grouped convolution
        in_zero_point: Input zero point
        weight_zero_point: Weight zero point
        bias_scale: Bias scale value
        output_scale: Output scale value
        output_zero_point: Output zero point
    
    Returns:
        dict: Dictionary containing calculated conv_params
    """
    # Calculate effective scale
    effective_scale = bias_scale / output_scale if output_scale != 0 else 0
    
    # Find the best output_shift so that outputScale fits in uint16_t
    best_shift = 15
    raw_scale = int(effective_scale * (1 << best_shift))
    
    if raw_scale > 65535:
        # Scale too large for uint16_t, reduce shift until it fits
        while best_shift > 0 and raw_scale > 65535:
            best_shift -= 1
            raw_scale = int(effective_scale * (1 << best_shift))
    elif raw_scale < 16384 and best_shift < 31:
        # Scale too small, increase shift for better precision
        while best_shift < 31:
            trial = int(effective_scale * (1 << (best_shift + 1)))
            if trial > 65535:
                break
            best_shift += 1
            raw_scale = trial
    
    # Clamp to valid uint16_t range [1, 65535]
    if raw_scale <= 0:
        raw_scale = 1
    if raw_scale > 65535:
        raw_scale = 65535
    
    return {
        'strideX': stride_w,
        'strideY': stride_h,
        'accumShift': 0,  # No pre-shift; keep full int32 accumulator precision
        'reluMax': 127,  # Max value for int8_t output
        'outputShift': best_shift,
        'outputScale': raw_scale,
        'dilation': max(dilation_h, dilation_w),
        'kernelHeight': wh,
        'kernelWidth': ww
    }


def main():
    """Main function with example usage."""
    # Configuration for 7x7j2d1 convolution
    # Input: n=1, c=3, h=224, w=224
    # Output: oc=64, wc=3, wh=7, ww=7, oh=112, ow=112
    conv_params_7x7 = calculate_conv_params(
        n=1, c=3, h=224, w=224,
        oc=64, wc=3, wh=7, ww=7,
        oh=112, ow=112,
        stride_h=2, stride_w=2,
        padding_h=3, padding_w=3,
        dilation_h=1, dilation_w=1,
        groups=1,
        in_zero_point=0,
        weight_zero_point=0,
        bias_scale=1.0,
        output_scale=1.0,
        output_zero_point=0
    )
    
    config_7x7 = {
        'input_whd': (224, 224, 3),
        'output_whd': (112, 112, 64),
        'kernel_whdn': (7, 7, 3, 64),
        'padding': (3, 3, 3, 3, 0, 0),
        'stride_xy': (2, 2),
        'kernel_name': "7x7j2d1",
        'data_type': "S8S8",
        'conv_params': (
            conv_params_7x7['strideX'],
            conv_params_7x7['strideY'],
            conv_params_7x7['accumShift'],
            conv_params_7x7['reluMax'],
            conv_params_7x7['outputShift'],
            conv_params_7x7['outputScale'],
            conv_params_7x7['dilation'],
            conv_params_7x7['kernelHeight'],
            conv_params_7x7['kernelWidth']
        ),
        'conv_flags': 0,
    }
    
    # Configuration for 3x3j1d1 convolution
    # Input: n=1, c=64, h=56, w=56
    # Output: oc=64, wc=64, wh=3, ww=3, oh=56, ow=56
    conv_params_3x3 = calculate_conv_params(
        n=1, c=64, h=56, w=56,
        oc=64, wc=64, wh=3, ww=3,
        oh=56, ow=56,
        stride_h=1, stride_w=1,
        padding_h=1, padding_w=1,
        dilation_h=1, dilation_w=1,
        groups=1,
        in_zero_point=0,
        weight_zero_point=0,
        bias_scale=1.0,
        output_scale=1.0,
        output_zero_point=0
    )
    
    config_3x3 = {
        'input_whd': (56, 56, 64),
        'output_whd': (56, 56, 64),
        'kernel_whdn': (3, 3, 64, 64),
        'padding': (1, 1, 1, 1, 0, 0),
        'stride_xy': (1, 1),
        'kernel_name': "3x3j1d1",
        'data_type': "S8S8",
        'conv_params': (
            conv_params_3x3['strideX'],
            conv_params_3x3['strideY'],
            conv_params_3x3['accumShift'],
            conv_params_3x3['reluMax'],
            conv_params_3x3['outputShift'],
            conv_params_3x3['outputScale'],
            conv_params_3x3['dilation'],
            conv_params_3x3['kernelHeight'],
            conv_params_3x3['kernelWidth']
        ),
        'conv_flags': 0,
    }
    
    # Configuration for 3x3j2d1 convolution
    # Input: n=1, c=64, h=56, w=56
    # Output: oc=128, wc=64, wh=3, ww=3, oh=28, ow=28
    conv_params_3x3j2d1 = calculate_conv_params(
        n=1, c=64, h=56, w=56,
        oc=128, wc=64, wh=3, ww=3,
        oh=28, ow=28,
        stride_h=2, stride_w=2,
        padding_h=1, padding_w=1,
        dilation_h=1, dilation_w=1,
        groups=1,
        in_zero_point=0,
        weight_zero_point=0,
        bias_scale=1.0,
        output_scale=1.0,
        output_zero_point=0
    )
    
    config_3x3j2d1 = {
        'input_whd': (56, 56, 64),
        'output_whd': (28, 28, 128),
        'kernel_whdn': (3, 3, 64, 128),
        'padding': (1, 1, 1, 1, 0, 0),
        'stride_xy': (2, 2),
        'kernel_name': "3x3j2d1",
        'data_type': "S8S8",
        'conv_params': (
            conv_params_3x3j2d1['strideX'],
            conv_params_3x3j2d1['strideY'],
            conv_params_3x3j2d1['accumShift'],
            conv_params_3x3j2d1['reluMax'],
            conv_params_3x3j2d1['outputShift'],
            conv_params_3x3j2d1['outputScale'],
            conv_params_3x3j2d1['dilation'],
            conv_params_3x3j2d1['kernelHeight'],
            conv_params_3x3j2d1['kernelWidth']
        ),
        'conv_flags': 0,
    }
    
    # Configuration for 1x1j2d1 convolution
    # Input: n=1, c=64, h=56, w=56
    # Output: oc=128, wc=64, wh=1, ww=1, oh=28, ow=28
    conv_params_1x1j2d1 = calculate_conv_params(
        n=1, c=64, h=56, w=56,
        oc=128, wc=64, wh=1, ww=1,
        oh=28, ow=28,
        stride_h=2, stride_w=2,
        padding_h=0, padding_w=0,
        dilation_h=1, dilation_w=1,
        groups=1,
        in_zero_point=0,
        weight_zero_point=0,
        bias_scale=1.0,
        output_scale=1.0,
        output_zero_point=0
    )
    
    config_1x1j2d1 = {
        'input_whd': (56, 56, 64),
        'output_whd': (28, 28, 128),
        'kernel_whdn': (1, 1, 64, 128),
        'padding': (0, 0, 0, 0, 0, 0),
        'stride_xy': (2, 2),
        'kernel_name': "1x1j2d1",
        'data_type': "S8S8",
        'conv_params': (
            conv_params_1x1j2d1['strideX'],
            conv_params_1x1j2d1['strideY'],
            conv_params_1x1j2d1['accumShift'],
            conv_params_1x1j2d1['reluMax'],
            conv_params_1x1j2d1['outputShift'],
            conv_params_1x1j2d1['outputScale'],
            conv_params_1x1j2d1['dilation'],
            conv_params_1x1j2d1['kernelHeight'],
            conv_params_1x1j2d1['kernelWidth']
        ),
        'conv_flags': 0,
    }
    
    # Configuration for 1x1j1d1 convolution
    # Input: n=1, c=512, h=28, w=28
    # Output: oc=256, wc=512, wh=1, ww=1, oh=28, ow=28
    conv_params_1x1j1d1 = calculate_conv_params(
        n=1, c=512, h=28, w=28,
        oc=256, wc=512, wh=1, ww=1,
        oh=28, ow=28,
        stride_h=1, stride_w=1,
        padding_h=0, padding_w=0,
        dilation_h=1, dilation_w=1,
        groups=1,
        in_zero_point=0,
        weight_zero_point=0,
        bias_scale=1.0,
        output_scale=1.0,
        output_zero_point=0
    )
    
    config_1x1j1d1 = {
        'input_whd': (28, 28, 512),
        'output_whd': (28, 28, 256),
        'kernel_whdn': (1, 1, 512, 256),
        'padding': (0, 0, 0, 0, 0, 0),
        'stride_xy': (1, 1),
        'kernel_name': "1x1j1d1",
        'data_type': "S8S8",
        'conv_params': (
            conv_params_1x1j1d1['strideX'],
            conv_params_1x1j1d1['strideY'],
            conv_params_1x1j1d1['accumShift'],
            conv_params_1x1j1d1['reluMax'],
            conv_params_1x1j1d1['outputShift'],
            conv_params_1x1j1d1['outputScale'],
            conv_params_1x1j1d1['dilation'],
            conv_params_1x1j1d1['kernelHeight'],
            conv_params_1x1j1d1['kernelWidth']
        ),
        'conv_flags': 0,
    }    
    
    
    # Find maximum configuration for 7x7j2d1
    best_n_tile_7x7, best_out_rows_7x7, buffer_sizes_7x7 = find_max_tile_config(**config_7x7)
    print_buffer_info(buffer_sizes_7x7)
    
    # Calculate and print buffer placement for 7x7
    placement_7x7 = calculate_buffer_placement(buffer_sizes_7x7, dram0_size=DRAM_SIZE_0, dram1_size=DRAM_SIZE_1)
    print_buffer_placement(placement_7x7)
    
    # Find maximum configuration for 3x3j1d1
    print("\n" + "="*60 + "\n")
    best_n_tile_3x3, best_out_rows_3x3, buffer_sizes_3x3 = find_max_tile_config(**config_3x3)
    print_buffer_info(buffer_sizes_3x3)
    
    # Calculate and print buffer placement for 3x3
    placement_3x3 = calculate_buffer_placement(buffer_sizes_3x3, dram0_size=DRAM_SIZE_0, dram1_size=DRAM_SIZE_1)
    print_buffer_placement(placement_3x3)
    
    # Find maximum configuration for 3x3j2d1
    print("\n" + "="*60 + "\n")
    best_n_tile_3x3j2d1, best_out_rows_3x3j2d1, buffer_sizes_3x3j2d1 = find_max_tile_config(**config_3x3j2d1)
    print_buffer_info(buffer_sizes_3x3j2d1)
    
    # Calculate and print buffer placement for 3x3j2d1
    placement_3x3j2d1 = calculate_buffer_placement(buffer_sizes_3x3j2d1, dram0_size=DRAM_SIZE_0, dram1_size=DRAM_SIZE_1)
    print_buffer_placement(placement_3x3j2d1)
    
    # Find maximum configuration for 1x1j2d1
    print("\n" + "="*60 + "\n")
    best_n_tile_1x1j2d1, best_out_rows_1x1j2d1, buffer_sizes_1x1j2d1 = find_max_tile_config(**config_1x1j2d1)
    print_buffer_info(buffer_sizes_1x1j2d1)
    
    # Calculate and print buffer placement for 1x1j2d1
    placement_1x1j2d1 = calculate_buffer_placement(buffer_sizes_1x1j2d1, dram0_size=DRAM_SIZE_0, dram1_size=DRAM_SIZE_1)
    print_buffer_placement(placement_1x1j2d1)
    
    # Find maximum configuration for 1x1j1d1
    print("\n" + "="*60 + "\n")
    best_n_tile_1x1j1d1, best_out_rows_1x1j1d1, buffer_sizes_1x1j1d1 = find_max_tile_config(**config_1x1j1d1)
    print_buffer_info(buffer_sizes_1x1j1d1)
    
    # Calculate and print buffer placement for 1x1j1d1
    placement_1x1j1d1 = calculate_buffer_placement(buffer_sizes_1x1j1d1, dram0_size=DRAM_SIZE_0, dram1_size=DRAM_SIZE_1)
    print_buffer_placement(placement_1x1j1d1)
    
    # Generate combined header content
    header_content = generate_combined_header([buffer_sizes_7x7, buffer_sizes_3x3, buffer_sizes_3x3j2d1, buffer_sizes_1x1j2d1, buffer_sizes_1x1j1d1])
    print("\n=== Generated Header Content ===")
    print(header_content)
    
    # Write to file
    output_file = r"C:\usr\xtensa\Xplorer-11.1.5-workspaces\xicnn1\test_cnn_depthwise_convolve_MOD2\test\convIdma_buffers.h"
    with open(output_file, 'w') as f:
        f.write(header_content)
    print(f"\nHeader file written to: {output_file}")


def generate_combined_header(buffer_sizes_list, header_guard="CONVIDMA_BUFFERS_H_", dram0_size=None, dram1_size=None):
    """
    Generate C header file content with buffer size definitions for multiple kernels.
    
    Args:
        buffer_sizes_list: List of dictionaries from calculate_buffer_sizes()
        header_guard: Header guard name
        dram0_size: Size of DRAM0 in bytes (default: use global DRAM_SIZE_0)
        dram1_size: Size of DRAM1 in bytes (default: use global DRAM_SIZE_1)
    
    Returns:
        String containing header file content
    """
    # Use global DRAM sizes if not specified
    if dram0_size is None:
        dram0_size = DRAM_SIZE_0
    if dram1_size is None:
        dram1_size = DRAM_SIZE_1
    
    header = f"""/*
 * convIdma_buffers.h
 *
 *  Auto-generated buffer size definitions
 */

#ifndef {header_guard}
#define {header_guard}


// ============================================================================
// Avilable DRAM Sizes for IDMA Buffers
// ============================================================================

#define IDMA_BUFFER_SIZE_DRAM0 ({dram0_size}) // {dram0_size // 1024} KB for DRAM0
#define IDMA_BUFFER_SIZE_DRAM1 ({dram1_size}) // {dram1_size // 1024} KB for DRAM1

"""
    
    # Calculate placements for all kernels
    placements = [calculate_buffer_placement(bs, dram0_size=dram0_size, dram1_size=dram1_size) 
                  for bs in buffer_sizes_list]
    
    # Generate content for each kernel configuration
    for buffer_sizes, placement in zip(buffer_sizes_list, placements):
        kernel_name = buffer_sizes['kernel_name']
        data_type = buffer_sizes['data_type']
        
        # Extract padding values
        dim1_edge1, dim1_edge2, dim2_edge1, dim2_edge2, dim3_edge1, dim3_edge2 = (
            buffer_sizes.get('padding', (0, 0, 0, 0, 0, 0))
        )
        
        header += f"""// ============================================================================
// IDMA Buffer Sizes and Tile Parameters for convVQ3D_{kernel_name}_{data_type}_MOW_WHD
// ============================================================================

// SRC tile parameters
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_SRC_DIM1_SIZE     {buffer_sizes['SRC_DIM1_SIZE']} // input width
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_SRC_DIM1_PITCH    {buffer_sizes['SRC_DIM1_PITCH']} //
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_SRC_DIM2_SIZE     {buffer_sizes['SRC_DIM2_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_SRC_DIM2_PITCH    {buffer_sizes['SRC_DIM2_PITCH']} // {buffer_sizes['SRC_DIM1_SIZE']}*{buffer_sizes['SRC_DIM2_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_SRC_DIM3_SIZE     {buffer_sizes['SRC_DIM3_SIZE']}

// DST   tile parameters
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_DST_DIM1_SIZE     {buffer_sizes['DST_DIM1_SIZE']} // input width
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_DST_DIM1_PITCH    {buffer_sizes['DST_DIM1_PITCH']} //
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_DST_DIM2_SIZE     {buffer_sizes['DST_DIM2_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_DST_DIM2_PITCH    {buffer_sizes['DST_DIM2_PITCH']} // {buffer_sizes['DST_DIM1_SIZE']}*{buffer_sizes['DST_DIM2_SIZE']}


// Input tile parameters
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_IN_DIM1_SIZE     {buffer_sizes['IN_DIM1_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_IN_DIM1_PITCH    {buffer_sizes['IN_DIM1_PITCH']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_IN_DIM2_SIZE     {buffer_sizes['IN_DIM2_SIZE']} // IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM2_SIZE + ((IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM2_SIZE-1)* stride)
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_IN_DIM2_PITCH    {buffer_sizes['IN_DIM2_PITCH']}

#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_IN_DIM1_EDGE1 {dim1_edge1}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_IN_DIM1_EDGE2 {dim1_edge2}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_IN_DIM2_EDGE1 {dim2_edge1}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_IN_DIM2_EDGE2 {dim2_edge2}  
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_IN_DIM3_EDGE1 {dim3_edge1}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_IN_DIM3_EDGE2 {dim3_edge2}


#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_IN_DATA_OFFSET   {buffer_sizes['IN_DATA_OFFSET']}
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_IN_ROWS_FIRSTDMA {buffer_sizes['IN_ROWS_FIRSTDMA']}  // IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_IN_DIM2_SIZE - padding rows 
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_IN_FRAME_PTR 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_IN_STATUS_FLAGS 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_IN_DIM1_COORD 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_IN_DIM3_COORD 0

// Output tile parameters
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM1_SIZE     {buffer_sizes['OUT_DIM1_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM1_PITCH    {buffer_sizes['OUT_DIM1_PITCH']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM2_SIZE     {buffer_sizes['OUT_DIM2_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM2_PITCH    {buffer_sizes['OUT_DIM2_PITCH']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM3_SIZE     {buffer_sizes['N_TILE_SIZE']} //IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_N_TILE_SIZE
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT_FRAME_PTR 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT_STATUS_FLAGS 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM1_COORD 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM1_EDGE1 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM1_EDGE2 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM2_EDGE1 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM2_EDGE2 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM3_COORD 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM3_EDGE1 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM3_EDGE2 0

//coefficient tile parameters
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM1_SIZE     {buffer_sizes['COEFF_DIM1_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM2_SIZE     {buffer_sizes['COEFF_DIM2_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM3_SIZE     {buffer_sizes['COEFF_DIM3_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM4_SIZE     {buffer_sizes['COEFF_DIM4_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM1_PITCH    {buffer_sizes['COEFF_DIM1_PITCH']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM2_PITCH    {buffer_sizes['COEFF_DIM2_PITCH']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM3_PITCH    {buffer_sizes['COEFF_DIM3_PITCH']}
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_FRAME_PTR 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_STATUS_FLAGS 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM1_COORD 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM1_EDGE1 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM1_EDGE2 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM2_COORD 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM2_EDGE1 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM2_EDGE2 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM3_COORD 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM3_EDGE1 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM3_EDGE2 0
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM4_COORD 0

//bias array parameters
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_BIAS_DIM1_SIZE       {buffer_sizes['BIAS_DIM1_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_BIAS_DIM2_SIZE       {buffer_sizes['BIAS_DIM2_SIZE']}       

//output scale array parameters
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_OUTSCALE_DIM1_SIZE     {buffer_sizes['OUTSCALE_DIM1_SIZE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_OUTSCALE_DIM2_SIZE     {buffer_sizes['OUTSCALE_DIM2_SIZE']}

// Buffer sizes
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_IN1       {buffer_sizes['IN']}
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_IN2       {buffer_sizes['IN']}
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF     {buffer_sizes['COEFF']}
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT1     {buffer_sizes['OUT']}
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT2     {buffer_sizes['OUT']}
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_BIAS       {buffer_sizes['BIAS']}
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUTSCALE   {buffer_sizes['OUTSCALE']}

"""
        
        # Generate DRAM placement macros based on optimization strategy
        dram_map = {}
        for name, size in placement['dram0_allocation']:
            dram_map[name] = 0
        for name, size in placement['dram1_allocation']:
            dram_map[name] = 1
        
        # Map buffer names to macro names
        header += f"""#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_IN1_DRAM     {dram_map.get('input_ping', 0)}  
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_IN2_DRAM     {dram_map.get('input_pong', 0)} 
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_COEFF_DRAM   {dram_map.get('coeff', 0)} 
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT1_DRAM     {dram_map.get('output_ping', 1)}
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUT2_DRAM     {dram_map.get('output_pong', 1)}
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_BIAS_DRAM     {dram_map.get('bias', 1)}
#define IDMA_BUFF_{kernel_name}_{data_type}_MOW_WHD_OUTSCALE_DRAM {dram_map.get('outscale', 1)}

#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_N_TILES {buffer_sizes['N_TILES']}           // round_toward positive(IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_SRC_DIM3_SIZE / IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_N_TILE_SIZE)
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_HIGHT_TILES {buffer_sizes['HIGHT_TILES']}     //IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_DST_DIM2_SIZE / IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_OUT_DIM2_SIZE   
#define  IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_N_TILE_SIZE {buffer_sizes['N_TILE_SIZE']}    // take this as input aas of now (contstant 22 for 3x3 conv and constant 64 for 7x7 conv)
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_N_TILE_SIZE_LAST {buffer_sizes['N_TILE_SIZE_LAST']} //IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM4_SIZE - IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_N_TILE_SIZE
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_TILE_SIZE_LAST {buffer_sizes['COEFF_TILE_SIZE_LAST']}  //  IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM1_SIZE * IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM2_SIZE * IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_COEFF_DIM3_SIZE * IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_N_TILE_SIZE_LAST
"""
        
        # Add convolution parameters if available
        if 'STRIDEX' in buffer_sizes:
            header += f"""
// Convolution parameters
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_STRIDEX {buffer_sizes['STRIDEX']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_STRIDEY {buffer_sizes['STRIDEY']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_ACCUM_SHIFT {buffer_sizes['ACCUM_SHIFT']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_RELU_MAX {buffer_sizes['RELU_MAX']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_RELU_MIN {buffer_sizes['RELU_MIN']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_OUTPUT_SHIFT {buffer_sizes['OUTPUT_SHIFT']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_OUTPUT_SCALE {buffer_sizes['OUTPUT_SCALE']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_DILATION {buffer_sizes['DILATION']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_KERNEL_HEIGHT {buffer_sizes['KERNEL_HEIGHT']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_KERNEL_WIDTH {buffer_sizes['KERNEL_WIDTH']}
#define IDMA_CONV_{kernel_name}_{data_type}_MOW_WHD_FLAGS {buffer_sizes['FLAGS']}
"""
        
        header += """

 
"""
    
    header += f"#endif /* {header_guard} */\n"
    return header


if __name__ == "__main__":
    main()
