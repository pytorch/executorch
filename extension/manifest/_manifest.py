from dataclasses import dataclass
from typing import ClassVar, Literal

from executorch.exir._serialize.padding import padding_required
from executorch.exir._warnings import experimental

# Byte order of numbers written to the manifest. Always little-endian
# regardless of the host system, since all commonly-used modern CPUs are little
# endian.
_MANIFEST_BYTEORDER: Literal["little"] = "little"


@dataclass
class _ManifestLayout:
    """Python class mirroring the binary layout of the manifest.
    separate from the Manifest class, which is the user facing
    representation.
    """

    EXPECTED_MAGIC: ClassVar[bytes] = b"em00"

    MAX_SIGNATURE_SIZE: ClassVar[int] = 512

    EXPECTED_MIN_LENGTH: ClassVar[int] = (
        # Header magic
        4
        # Header length
        + 4
        # program offset
        + 8
        # Padding
        + 4
        # signature size
        + 4
    )

    EXPECTED_MAX_LENGTH: ClassVar[int] = EXPECTED_MIN_LENGTH + MAX_SIGNATURE_SIZE

    signature: bytes

    # The actual size of the signature
    signature_size: int = 0

    # The size of any padding required
    padding_size: int = 0

    # Size in bytes between the top of the manifest and the start of the data it was appended to.
    program_offset: int = 0

    # The manifest length, in bytes, read from or to be written to the binary
    # footer.
    length: int = 0

    # The magic bytes read from or to be written to the binary footer.
    magic: bytes = EXPECTED_MAGIC

    def __post_init__(self):
        """Post init hook to validate the manifest."""
        if self.signature_size == 0:
            self.signature_size = len(self.signature)
        if self.length == 0:
            self.length = _ManifestLayout.EXPECTED_MIN_LENGTH + self.signature_size

        # Not using self.is_valid() here to deliver better error messages.
        if len(self.signature) > _ManifestLayout.MAX_SIGNATURE_SIZE:
            raise ValueError(
                f"Signature is too large. {self.signature_size}. Manifest only supports signatures up to {_ManifestLayout.MAX_SIGNATURE_SIZE} bytes."
            )
        if self.magic != _ManifestLayout.EXPECTED_MAGIC:
            raise ValueError(
                f"Invalid magic. Expected {_ManifestLayout.EXPECTED_MAGIC}. Got {self.magic}"
            )
        if self.length < _ManifestLayout.EXPECTED_MIN_LENGTH:
            raise ValueError(
                f"Invalid length. Expected at least {_ManifestLayout.EXPECTED_MIN_LENGTH}. Got {self.length}"
            )
        if self.length > _ManifestLayout.EXPECTED_MAX_LENGTH:
            raise ValueError(
                f"Invalid length. Expected at most {_ManifestLayout.EXPECTED_MAX_LENGTH}. Got {self.length}"
            )
        if self.signature_size != len(self.signature):
            raise ValueError(
                f"Invalid signature size must match len(self.signature). Expected {len(self.signature)}. Got {self.signature_size}"
            )

    def is_valid(self) -> bool:
        """Returns true if the manifest appears to be well-formed."""
        return (
            self.magic == _ManifestLayout.EXPECTED_MAGIC
            and self.length >= _ManifestLayout.EXPECTED_MIN_LENGTH
            and self.length <= _ManifestLayout.EXPECTED_MAX_LENGTH
            and self.signature_size >= 0
            and self.signature_size <= _ManifestLayout.MAX_SIGNATURE_SIZE
            and self.program_offset >= 0
            and len(self.signature) == self.signature_size
        )

    def to_bytes(self) -> bytes:
        """"Returns the binary representation of the Manifest. Written bottom up
        to allow for BC considerations. The compatibility-preserving way to make
        changes is to increase the header's length field and add new fields at
        the top. This means we can always check the last 8 bytes for the magic
        and size, and then load the full footer."
        """
        if not self.is_valid():
            raise ValueError("Cannot serialize an invalid manifest")

        data: bytes = (
            # bytes: Signature unique ID for the data the manifest was appended to.
            self.signature
            # actual size of the signature
            + self.signature_size.to_bytes(4, byteorder=_MANIFEST_BYTEORDER)
            # uint32_t: Any padding required to align the manifest.
            + self.padding_size.to_bytes(4, byteorder=_MANIFEST_BYTEORDER)
            # uint64_t: Size in bytes between the manifest and the data it was appended to.
            + self.program_offset.to_bytes(8, byteorder=_MANIFEST_BYTEORDER)
            # uint32_t: Actual size of this manifest.
            + self.length.to_bytes(4, byteorder=_MANIFEST_BYTEORDER)
            # Manifest magic. This lets consumers detect whether the
            # manifest was inserted or not. Always use the proper magic value
            # (i.e., ignore self.magic) since there's no reason to create an
            # invalid manifest.
            + self.EXPECTED_MAGIC
        )
        return data

    @staticmethod
    def from_bytes(data: bytes) -> "_ManifestLayout":
        """Tries to read a manifest from the provided data.

        Does not validate that the header is well-formed. Callers should
        use is_valid().

        Args:
            data: The data to read from.
        Returns:
            The contents of the serialized manifest.
        Raises:
            ValueError: If not enough data is provided.
        """
        if len(data) <= _ManifestLayout.EXPECTED_MIN_LENGTH:
            raise ValueError(
                f"Not enough data for the manifest: {len(data)} "
                + f"< {_ManifestLayout.EXPECTED_MIN_LENGTH}"
            )
        magic = data[-4:]
        length = int.from_bytes(data[-8:-4], byteorder=_MANIFEST_BYTEORDER)
        program_offset = int.from_bytes(data[-16:-8], byteorder=_MANIFEST_BYTEORDER)
        padding_size = int.from_bytes(data[-20:-16], byteorder=_MANIFEST_BYTEORDER)
        signature_size = int.from_bytes(data[-24:-20], byteorder=_MANIFEST_BYTEORDER)
        signature = data[-(signature_size + 24) : -24]
        return _ManifestLayout(
            signature=signature,
            signature_size=signature_size,
            padding_size=padding_size,
            program_offset=program_offset,
            length=length,
            magic=magic,
        )

    @staticmethod
    def from_manifest(manifest: "Manifest") -> "_ManifestLayout":
        return _ManifestLayout(
            signature=manifest.signature,
            signature_size=len(manifest.signature),
            length=_ManifestLayout.EXPECTED_MIN_LENGTH + len(manifest.signature),
            # program_offset and padding_size are set at append time.
        )


@experimental("This API is experimental and subject to change without notice.")
@dataclass
class Manifest:
    """A manifest that can be appended to a binary blob. The manifest contains
    meta information about the binary blob. You must know who created the manifest
    to be able to interpret the data in the manifest."""

    # Unique ID for the data the manifest was appended to. Often this might contain
    # a crytographic signature for the data.
    signature: bytes

    @staticmethod
    def _from_manifest_layout(layout: _ManifestLayout) -> "Manifest":
        return Manifest(
            signature=layout.signature,
        )

    @staticmethod
    def from_bytes(data: bytes) -> "Manifest":
        """Tries to read a manifest from the provided data."""
        layout = _ManifestLayout.from_bytes(data)
        if not layout.is_valid():
            raise ValueError("Cannot parse manifest from bytes")
        return Manifest._from_manifest_layout(layout)


@experimental("This API is experimental and subject to change without notice.")
def append_manifest(pte_data: bytes, manifest: Manifest, alignment: int = 16):
    """Appends a manifest to the provided data."""
    padding = padding_required(len(pte_data), alignment)

    manifest_layout = _ManifestLayout.from_manifest(manifest)
    manifest_layout.program_offset = len(pte_data) + manifest_layout.padding_size
    manifest_layout.padding_size = padding

    return pte_data + (b"\x00" * padding) + manifest_layout.to_bytes()
