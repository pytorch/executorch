# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict


def pad_to(data: bytes, length: int) -> bytes:
    """Returns the input followed by enough zero bytes to become the requested length.

    Args:
        data: The data to pad.
        length: The length of the returned data.
    Returns:
        The padded data.
    Raises:
        ValueError: If the requested length is less than the input length.
    """
    if length < len(data):
        raise ValueError(f"Data length {len(data)} > padded length {length}")
    if length > len(data):
        data = data + b"\x00" * (length - len(data))
    assert len(data) == length
    return data


def padding_required(offset: int, alignment: int) -> int:
    """Returns the padding required to align `offset` to `alignment`."""
    remainder: int = offset % alignment
    if remainder != 0:
        return alignment - remainder
    return 0


def aligned_size(input_size: int, alignment: int) -> int:
    """Returns input_size padded up to the next whole multiple of alignment."""
    return input_size + padding_required(input_size, alignment)
