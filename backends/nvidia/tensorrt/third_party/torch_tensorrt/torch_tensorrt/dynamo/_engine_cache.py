import copy
import io
import logging
import os
import pickle
import pickletools
import shutil
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch._inductor.codecache import sha256_hash
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._settings import (
    _SETTINGS_TO_BE_ENGINE_INVARIANT,
    CompilationSettings,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)

UnpackedCacheHit = Tuple[
    bytes,
    List[str],
    List[str],
    Sequence[Input],
    CompilationSettings,
    Optional[Dict[str, Any]],
    bool,
]


class BaseEngineCache(ABC):
    @abstractmethod
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass

    @staticmethod
    def get_hash(
        gm: torch.fx.GraphModule,
        input_specs: Sequence[Input],
        settings: CompilationSettings,
    ) -> str:
        """Get the hash value of the GraphModule

        Args:
            gm (torch.fx.GraphModule): GraphModule to hash
            input_specs (Sequence[Input]): input specs for the GraphModule
            settings (CompilationSettings): compilation settings for the GraphModule

        Returns:
            str: hash value of the GraphModule
        """

        def canonicalize_graph(graph: torch.fx.Graph) -> str:
            """Canonicalize the graph to a string for isomorphic graph comparison

            Args:
                graph (torch.fx.Graph): graph to canonicalize

            Returns:
                str: canonicalized graph string
            """
            canonical_nodes = []
            input_counter = 0

            for node in graph.nodes:
                if node.op == "placeholder":
                    canonical_nodes.append(f"placeholder_input_{input_counter}")
                    input_counter += 1
                else:
                    canonical_nodes.append(f"{node.op}_{node.target}")

            return " ".join(canonical_nodes)

        graph_str = canonicalize_graph(gm.graph)
        _LOGGER.debug(f"graph_str:\n {graph_str}")

        graph_hash = sha256_hash(graph_str.encode())

        input_spec_strs = [str(i) for i in input_specs]
        with io.BytesIO() as stream:
            input_specs_data = pickle.dumps(input_spec_strs)
            input_specs_data = pickletools.optimize(input_specs_data)
        input_specs_hash = sha256_hash(input_specs_data)

        invariant_engine_specs = [
            str(getattr(settings, field))
            for field in sorted(_SETTINGS_TO_BE_ENGINE_INVARIANT)
        ]
        with io.BytesIO() as stream:
            engine_specs_data = pickle.dumps(invariant_engine_specs)
            engine_specs_data = pickletools.optimize(engine_specs_data)
        engine_specs_hash = sha256_hash(engine_specs_data)

        hash_val: str = graph_hash + input_specs_hash + engine_specs_hash

        return hash_val

    @staticmethod
    def pack(
        serialized_engine: bytes,
        input_names: List[str],
        output_names: List[str],
        input_specs: Sequence[Input],
        compilation_settings: CompilationSettings,
        weight_name_map: Optional[Dict[Any, Any]],
        requires_output_allocator: bool,
    ) -> bytes:
        """Pack serialized engine, input names, output names, and weight map into a single blob

        Args:
            serialized_engine (bytes): serialized TRT engine
            input_names (List[str]): input names of TRT engine
            output_names (List[str]): output names of TRT engine
            input_specs (Sequence[Input]): input specs of TRT engine
            compilation_settings (CompilationSettings): compilation settings of TRT engine
            weight_name_map (Optional[Dict[Any, Any]]): weight name map for refitting
            requires_output_allocator (bool): Boolean flag indicating if the converter creates operators which require an Output Allocator to run (e.g. data dependent operators)
        Returns:
            bytes: packed blob
        """

        settings = copy.deepcopy(compilation_settings)
        return pickle.dumps(
            {
                "serialized_engine": bytes(serialized_engine),
                "input_names": input_names,
                "output_names": output_names,
                "input_specs": input_specs,
                "compilation_settings": settings,
                "weight_name_map": weight_name_map,
                "requires_output_allocator": requires_output_allocator,
            }
        )

    @staticmethod
    def unpack(packed_obj: bytes) -> UnpackedCacheHit:
        """Unpack packed blob into serialized engine, input names, output names, and weight map

        Args:
            packed_obj (bytes): packed blob

        Returns:
            Tuple[bytes, List[str], List[str], Sequence[Input], CompilationSettings, Optional[Dict[str, Any]]]: serialized engine, input names, output names, input specs, CompilationSettings, weight name map
        """
        unpacked = pickle.loads(packed_obj)
        return (
            unpacked["serialized_engine"],
            unpacked["input_names"],
            unpacked["output_names"],
            unpacked["input_specs"],
            unpacked["compilation_settings"],
            unpacked["weight_name_map"],
            unpacked["requires_output_allocator"],
        )

    def insert(
        self, hash: str, entry: UnpackedCacheHit, *args: Any, **kwargs: Any
    ) -> None:
        """
        Insert a cache entry into the engine cache.

        Args:
            hash (str): The hash value of the GraphModule.
            entry (Tuple[bytes, List[str], List[str], CompilationSettings, Optional[Dict[Any, Any]]]): The cache entry to be inserted.
            *args: Variable length argument list passed to ``save``.
            **kwargs: Arbitrary keyword arguments passed to ``save``.

        Returns:
            None
        """
        packed_cache_info = BaseEngineCache.pack(*entry)
        return self.save(hash, packed_cache_info, *args, **kwargs)

    def check(self, hash: str, *args: Any, **kwargs: Any) -> Optional[UnpackedCacheHit]:
        """
        Check if a cache entry exists for the given hash.

        Args:
            hash (str): The hash value of the GraphModule.
            *args: Variable length argument list passed to ``load``.
            **kwargs: Arbitrary keyword arguments passed to ``load``.

        Returns:
            Optional[Tuple[bytes, List[str], List[str], CompilationSettings, Optional[Dict[Any, Any]]]]: The unpacked cache entry if found, None otherwise.
        """
        packed_cache_info = self.load(hash, *args, **kwargs)

        if packed_cache_info:
            return BaseEngineCache.unpack(packed_cache_info)
        else:
            return None

    @abstractmethod
    def save(self, hash: str, blob: bytes, *args: Any, **kwargs: Any) -> None:
        """Store blob in cache

        Args:
            hash (str): hash value of the GraphModule
            blob (bytes): packed blob
        """
        pass

    @abstractmethod
    def load(self, hash: str, *args: Any, **kwargs: Any) -> Optional[bytes]:
        """Load blob from storage

        Args:
            hash (str): hash value of the GraphModule

        Returns:
            Optional[bytes]: blob or None if doesn't hit
        """
        pass


class DiskEngineCache(BaseEngineCache):
    dir2hash2size_map: Dict[str, Dict[str, int]] = (
        {}
    )  # dir2hash2size_map["engine_cache_dir"]["hash"] = size

    def __init__(
        self,
        engine_cache_dir: str,
        engine_cache_size: int,
    ) -> None:
        def get_dir_size(path: str) -> int:
            total = 0
            with os.scandir(path) as it:
                for entry in it:
                    if entry.is_file():
                        total += entry.stat().st_size
                    elif entry.is_dir():
                        total += get_dir_size(entry.path)
            return total

        if not os.path.exists(engine_cache_dir):
            os.makedirs(engine_cache_dir, exist_ok=True)
        self.engine_cache_dir = engine_cache_dir
        self.total_engine_cache_size = engine_cache_size
        self.available_engine_cache_size = engine_cache_size - get_dir_size(
            engine_cache_dir
        )
        if engine_cache_dir not in DiskEngineCache.dir2hash2size_map:
            DiskEngineCache.dir2hash2size_map[engine_cache_dir] = {}

        _LOGGER.info(
            f"Disk engine cache initialized (cache directory:{self.engine_cache_dir}, max size: {self.total_engine_cache_size})"
        )

    def has_available_cache_size(self, needed_size: int) -> bool:
        """Check if the cache has available space for saving object

        Args:
            needed_size (int): needed size for saving object

        Returns:
            bool: whether the cache has available size for saving object
        """
        return needed_size <= self.available_engine_cache_size

    def clear_cache(self, needed_min_size: int) -> None:
        """Clear the cache to make sure at least `needed_min_size` bytes are available, if possible

        Args:
            needed_min_size (int): the minimum needed size
        """

        def LRU() -> None:
            """Clear the Least Recently Used engine in the cache"""
            # Get the list of engine directories
            engines_hash_values = os.listdir(self.engine_cache_dir)
            # Sort the engine directories by modification time (oldest first)
            engines_hash_values.sort(
                key=lambda x: os.path.getmtime(os.path.join(self.engine_cache_dir, x))
            )
            # Iterate over the engine directories and remove the oldest ones until enough space is available
            for engine_hash in engines_hash_values:
                if self.available_engine_cache_size >= needed_min_size:
                    break
                engine_path = os.path.join(self.engine_cache_dir, engine_hash)
                try:
                    # Remove the entire directory
                    shutil.rmtree(engine_path)
                    # Update the available cache size
                    self.available_engine_cache_size += (
                        DiskEngineCache.dir2hash2size_map[self.engine_cache_dir].pop(
                            engine_hash, 0
                        )
                    )
                    _LOGGER.debug(
                        f"Removed the engine cache at {engine_path}, available cache size: {self.available_engine_cache_size} bytes."
                    )
                except Exception as e:
                    _LOGGER.warning(
                        f"Failed to clear the engine cache at {engine_path}: {e}"
                    )

        if needed_min_size > self.total_engine_cache_size:
            _LOGGER.warning(
                f"The needed minimum size {needed_min_size} is larger than the total cache size {self.total_engine_cache_size}. Nothing will be cleared."
            )
        else:
            LRU()

    def save(self, hash: str, blob: bytes, *args: Any, **kwargs: Any) -> None:
        blob_size = len(blob)
        if blob_size > self.total_engine_cache_size:
            _LOGGER.warning(
                f"The serialized engine cannot be saved because the size {blob_size} is larger than the total cache size {self.total_engine_cache_size}."
            )
            return

        if not self.has_available_cache_size(blob_size):
            self.clear_cache(blob_size)

        if self.has_available_cache_size(blob_size):
            DiskEngineCache.dir2hash2size_map[self.engine_cache_dir][hash] = blob_size
            self.available_engine_cache_size -= blob_size
            directory = os.path.join(self.engine_cache_dir, hash)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            blob_path = os.path.join(
                directory,
                "blob.bin",
            )
            try:
                with open(blob_path, "wb") as f:
                    f.write(blob)
                _LOGGER.debug(f"The engine added to cache, saved to {blob_path}")
            except Exception as e:
                del DiskEngineCache.dir2hash2size_map[self.engine_cache_dir][hash]
                self.available_engine_cache_size += blob_size
                shutil.rmtree(directory)
                _LOGGER.warning(f"Failed to save the blob to {blob_path}: {e}")

        else:
            _LOGGER.warning(
                f"The size {blob_size} is still larger than the available cache size {self.available_engine_cache_size}."
            )

    def load(self, hash: str, *args: Any, **kwargs: Any) -> Optional[bytes]:
        directory = os.path.join(self.engine_cache_dir, hash)
        if os.path.exists(directory):
            blob_path = os.path.join(directory, "blob.bin")
            if os.path.exists(blob_path):
                with open(blob_path, "rb") as f:
                    blob = f.read()
                _LOGGER.debug(f"Engine found in cache, loaded from {blob_path}")
                return blob
        return None
