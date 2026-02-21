# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest
from unittest.mock import MagicMock

import torch
from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
    VkMemoryLayout,
    VkStorageType,
)
from executorch.backends.vulkan.utils import (
    ANY_BUFFER,
    ANY_STORAGE,
    ANY_TEXTURE,
    CHANNELS_PACKED_ANY,
    CHANNELS_PACKED_TEXTURE,
    CONTIGUOUS_ANY,
    CONTIGUOUS_BUFFER,
    DEFAULT_TEXTURE_LIMITS,
    HEIGHT_PACKED_TEXTURE,
    make_tensor_repset,
    NO_STORAGE,
    OpRepSets,
    PACKED_INT8_4C1W_BUFFER,
    PACKED_INT8_4W4C_BUFFER,
    PACKED_INT8_4W_BUFFER,
    PACKED_INT8_BUFFER,
    PACKED_INT8_CHANNELS_PACKED_BUFFER,
    TensorRepr,
    TensorReprList,
    TensorRepSet,
    TensorRepSetList,
    WIDTH_PACKED_TEXTURE,
)
from torch._subclasses.fake_tensor import FakeTensorMode


def _make_fake_tensor(shape, dtype=torch.float32):
    with FakeTensorMode() as mode:
        return mode.from_tensor(torch.empty(shape, dtype=dtype))


def _make_op_node(
    target,
    args,
    output_val,
):
    """Create a mock torch.fx.Node for use in OpRepSets tests."""
    node = MagicMock(spec=torch.fx.Node)
    node.op = "call_function"
    node.target = target
    node.args = args
    node.meta = {"val": output_val}
    return node


def _make_tensor_arg_node(shape, dtype=torch.float32):
    """Create a mock arg node that looks like a single tensor node."""
    node = MagicMock(spec=torch.fx.Node)
    node.op = "call_function"
    fake = _make_fake_tensor(shape, dtype)
    node.meta = {"val": fake}
    return node


class TestTensorRepSet(unittest.TestCase):
    # -- Construction and emptiness --

    def test_empty_repset(self):
        repset = TensorRepSet(set(), set())
        self.assertTrue(repset.is_empty())
        self.assertFalse(repset.texture_is_valid())
        self.assertFalse(repset.buffer_is_valid())

    def test_non_empty_repset(self):
        repset = TensorRepSet(
            {VkMemoryLayout.TENSOR_WIDTH_PACKED},
            {VkMemoryLayout.TENSOR_CHANNELS_PACKED},
        )
        self.assertFalse(repset.is_empty())
        self.assertTrue(repset.texture_is_valid())
        self.assertTrue(repset.buffer_is_valid())

    def test_texture_only_repset(self):
        repset = TensorRepSet(set(), {VkMemoryLayout.TENSOR_CHANNELS_PACKED})
        self.assertFalse(repset.is_empty())
        self.assertTrue(repset.texture_is_valid())
        self.assertFalse(repset.buffer_is_valid())

    def test_buffer_only_repset(self):
        repset = TensorRepSet({VkMemoryLayout.TENSOR_WIDTH_PACKED}, set())
        self.assertFalse(repset.is_empty())
        self.assertFalse(repset.texture_is_valid())
        self.assertTrue(repset.buffer_is_valid())

    # -- Equality --

    def test_equality(self):
        a = TensorRepSet(
            {VkMemoryLayout.TENSOR_WIDTH_PACKED},
            {VkMemoryLayout.TENSOR_CHANNELS_PACKED},
        )
        b = TensorRepSet(
            {VkMemoryLayout.TENSOR_WIDTH_PACKED},
            {VkMemoryLayout.TENSOR_CHANNELS_PACKED},
        )
        self.assertEqual(a, b)

    def test_inequality(self):
        a = TensorRepSet(
            {VkMemoryLayout.TENSOR_WIDTH_PACKED},
            {VkMemoryLayout.TENSOR_CHANNELS_PACKED},
        )
        b = TensorRepSet(
            {VkMemoryLayout.TENSOR_HEIGHT_PACKED},
            {VkMemoryLayout.TENSOR_CHANNELS_PACKED},
        )
        self.assertNotEqual(a, b)

    # -- Copy --

    def test_copy_produces_equal_repset(self):
        repset = TensorRepSet(
            {VkMemoryLayout.TENSOR_WIDTH_PACKED},
            {VkMemoryLayout.TENSOR_CHANNELS_PACKED},
        )
        copied = repset.copy()
        self.assertEqual(repset, copied)

    def test_copy_is_independent(self):
        repset = TensorRepSet(
            {VkMemoryLayout.TENSOR_WIDTH_PACKED},
            {VkMemoryLayout.TENSOR_CHANNELS_PACKED},
        )
        copied = repset.copy()
        copied.valid_buffer_layouts.add(VkMemoryLayout.TENSOR_HEIGHT_PACKED)
        self.assertNotEqual(repset, copied)
        self.assertNotIn(
            VkMemoryLayout.TENSOR_HEIGHT_PACKED, repset.valid_buffer_layouts
        )

    # -- Intersection --

    def test_make_intersect(self):
        a = TensorRepSet(
            {VkMemoryLayout.TENSOR_WIDTH_PACKED, VkMemoryLayout.TENSOR_HEIGHT_PACKED},
            {VkMemoryLayout.TENSOR_CHANNELS_PACKED, VkMemoryLayout.TENSOR_WIDTH_PACKED},
        )
        b = TensorRepSet(
            {VkMemoryLayout.TENSOR_WIDTH_PACKED},
            {VkMemoryLayout.TENSOR_CHANNELS_PACKED},
        )
        result = a.make_intersect(b)
        self.assertEqual(
            result.valid_buffer_layouts, {VkMemoryLayout.TENSOR_WIDTH_PACKED}
        )
        self.assertEqual(
            result.valid_texture_layouts, {VkMemoryLayout.TENSOR_CHANNELS_PACKED}
        )

    def test_make_intersect_disjoint_yields_empty(self):
        a = TensorRepSet(
            {VkMemoryLayout.TENSOR_WIDTH_PACKED}, {VkMemoryLayout.TENSOR_WIDTH_PACKED}
        )
        b = TensorRepSet(
            {VkMemoryLayout.TENSOR_HEIGHT_PACKED},
            {VkMemoryLayout.TENSOR_HEIGHT_PACKED},
        )
        result = a.make_intersect(b)
        self.assertTrue(result.is_empty())

    # -- Union --

    def test_make_union(self):
        a = TensorRepSet(
            {VkMemoryLayout.TENSOR_WIDTH_PACKED}, {VkMemoryLayout.TENSOR_WIDTH_PACKED}
        )
        b = TensorRepSet(
            {VkMemoryLayout.TENSOR_HEIGHT_PACKED},
            {VkMemoryLayout.TENSOR_CHANNELS_PACKED},
        )
        result = a.make_union(b)
        self.assertEqual(
            result.valid_buffer_layouts,
            {VkMemoryLayout.TENSOR_WIDTH_PACKED, VkMemoryLayout.TENSOR_HEIGHT_PACKED},
        )
        self.assertEqual(
            result.valid_texture_layouts,
            {VkMemoryLayout.TENSOR_WIDTH_PACKED, VkMemoryLayout.TENSOR_CHANNELS_PACKED},
        )

    # -- Compatibility checks --

    def test_is_compatible_texture(self):
        repset = TensorRepSet(set(), {VkMemoryLayout.TENSOR_CHANNELS_PACKED})
        tr = TensorRepr(VkStorageType.TEXTURE_3D, VkMemoryLayout.TENSOR_CHANNELS_PACKED)
        self.assertTrue(repset.is_compatible(tr))

    def test_is_compatible_texture_mismatch(self):
        repset = TensorRepSet(set(), {VkMemoryLayout.TENSOR_CHANNELS_PACKED})
        tr = TensorRepr(VkStorageType.TEXTURE_3D, VkMemoryLayout.TENSOR_WIDTH_PACKED)
        self.assertFalse(repset.is_compatible(tr))

    def test_is_compatible_buffer(self):
        repset = TensorRepSet({VkMemoryLayout.TENSOR_WIDTH_PACKED}, set())
        tr = TensorRepr(VkStorageType.BUFFER, VkMemoryLayout.TENSOR_WIDTH_PACKED)
        self.assertTrue(repset.is_compatible(tr))

    def test_is_compatible_buffer_mismatch(self):
        repset = TensorRepSet({VkMemoryLayout.TENSOR_WIDTH_PACKED}, set())
        tr = TensorRepr(VkStorageType.BUFFER, VkMemoryLayout.TENSOR_HEIGHT_PACKED)
        self.assertFalse(repset.is_compatible(tr))

    # -- any_in_common --

    def test_any_in_common_true(self):
        a = TensorRepSet(
            {VkMemoryLayout.TENSOR_WIDTH_PACKED},
            {VkMemoryLayout.TENSOR_CHANNELS_PACKED},
        )
        b = TensorRepSet(
            {VkMemoryLayout.TENSOR_WIDTH_PACKED},
            {VkMemoryLayout.TENSOR_WIDTH_PACKED},
        )
        self.assertTrue(a.any_in_common(b))

    def test_any_in_common_false(self):
        a = TensorRepSet(
            {VkMemoryLayout.TENSOR_WIDTH_PACKED}, {VkMemoryLayout.TENSOR_WIDTH_PACKED}
        )
        b = TensorRepSet(
            {VkMemoryLayout.TENSOR_HEIGHT_PACKED},
            {VkMemoryLayout.TENSOR_HEIGHT_PACKED},
        )
        self.assertFalse(a.any_in_common(b))

    # -- Constrained / Ambiguous --

    def test_is_constrained_empty(self):
        self.assertTrue(NO_STORAGE.is_constrained())

    def test_is_constrained_single_texture(self):
        repset = TensorRepSet(set(), {VkMemoryLayout.TENSOR_CHANNELS_PACKED})
        self.assertTrue(repset.is_constrained())

    def test_is_constrained_single_buffer(self):
        repset = TensorRepSet({VkMemoryLayout.TENSOR_WIDTH_PACKED}, set())
        self.assertTrue(repset.is_constrained())

    def test_is_ambiguous_multiple_layouts(self):
        repset = TensorRepSet(
            {VkMemoryLayout.TENSOR_WIDTH_PACKED, VkMemoryLayout.TENSOR_HEIGHT_PACKED},
            set(),
        )
        self.assertTrue(repset.is_ambiguous())

    def test_is_ambiguous_both_storage_types(self):
        repset = TensorRepSet(
            {VkMemoryLayout.TENSOR_WIDTH_PACKED},
            {VkMemoryLayout.TENSOR_CHANNELS_PACKED},
        )
        self.assertTrue(repset.is_ambiguous())

    # -- make_tensor_repr --

    def test_make_tensor_repr_prefers_texture(self):
        repset = TensorRepSet(
            {VkMemoryLayout.TENSOR_WIDTH_PACKED},
            {VkMemoryLayout.TENSOR_CHANNELS_PACKED},
        )
        tr = repset.make_tensor_repr()
        self.assertEqual(tr.storage_type, VkStorageType.TEXTURE_3D)
        self.assertEqual(tr.memory_layout, VkMemoryLayout.TENSOR_CHANNELS_PACKED)

    def test_make_tensor_repr_falls_back_to_buffer(self):
        repset = TensorRepSet({VkMemoryLayout.TENSOR_WIDTH_PACKED}, set())
        tr = repset.make_tensor_repr()
        self.assertEqual(tr.storage_type, VkStorageType.BUFFER)
        self.assertEqual(tr.memory_layout, VkMemoryLayout.TENSOR_WIDTH_PACKED)

    def test_make_tensor_repr_empty_returns_default(self):
        tr = NO_STORAGE.make_tensor_repr()
        self.assertEqual(tr.storage_type, VkStorageType.DEFAULT_STORAGE)
        self.assertEqual(tr.memory_layout, VkMemoryLayout.DEFAULT_LAYOUT)

    # -- has_same_packed_dim_info_set --

    def test_has_same_packed_dim_info_set(self):
        self.assertTrue(
            CHANNELS_PACKED_TEXTURE.has_same_packed_dim_info_set(
                CHANNELS_PACKED_TEXTURE
            )
        )
        self.assertTrue(
            PACKED_INT8_4W4C_BUFFER.has_same_packed_dim_info_set(
                PACKED_INT8_4C1W_BUFFER
            )
        )
        self.assertTrue(
            PACKED_INT8_BUFFER.has_same_packed_dim_info_set(PACKED_INT8_BUFFER)
        )
        self.assertFalse(
            PACKED_INT8_BUFFER.has_same_packed_dim_info_set(PACKED_INT8_4C1W_BUFFER)
        )

    def test_has_same_packed_dim_info_set_empty_is_compatible(self):
        self.assertTrue(
            NO_STORAGE.has_same_packed_dim_info_set(CHANNELS_PACKED_TEXTURE)
        )
        self.assertTrue(
            CHANNELS_PACKED_TEXTURE.has_same_packed_dim_info_set(NO_STORAGE)
        )
        self.assertTrue(NO_STORAGE.has_same_packed_dim_info_set(NO_STORAGE))

    def test_has_same_packed_dim_info_set_different_texture_layouts(self):
        self.assertFalse(
            WIDTH_PACKED_TEXTURE.has_same_packed_dim_info_set(CHANNELS_PACKED_TEXTURE)
        )

    def test_has_same_packed_dim_info_set_different_storage_types(self):
        # CHANNELS_PACKED_ANY has both buffer and texture layouts,
        # CHANNELS_PACKED_TEXTURE has only texture layouts
        self.assertFalse(
            CHANNELS_PACKED_ANY.has_same_packed_dim_info_set(CHANNELS_PACKED_TEXTURE)
        )

    def test_has_same_packed_dim_info_set_any_storage_self_compatible(self):
        self.assertTrue(ANY_STORAGE.has_same_packed_dim_info_set(ANY_STORAGE))

    # -- has_compatible_packed_dim_info_set --

    def test_has_compatible_packed_dim_info_set_self(self):
        self.assertTrue(
            CHANNELS_PACKED_TEXTURE.has_compatible_packed_dim_info_set(
                CHANNELS_PACKED_TEXTURE
            )
        )

    def test_has_compatible_packed_dim_info_set_superset(self):
        # ANY_TEXTURE has all packed dims, so it's a superset of any single layout
        self.assertTrue(
            ANY_TEXTURE.has_compatible_packed_dim_info_set(CHANNELS_PACKED_TEXTURE)
        )
        self.assertTrue(
            ANY_TEXTURE.has_compatible_packed_dim_info_set(WIDTH_PACKED_TEXTURE)
        )

    def test_has_compatible_packed_dim_info_set_subset_fails(self):
        # A single layout is not a superset of all layouts
        self.assertFalse(
            CHANNELS_PACKED_TEXTURE.has_compatible_packed_dim_info_set(ANY_TEXTURE)
        )

    def test_has_compatible_packed_dim_info_set_disjoint(self):
        self.assertFalse(
            WIDTH_PACKED_TEXTURE.has_compatible_packed_dim_info_set(
                CHANNELS_PACKED_TEXTURE
            )
        )

    def test_has_compatible_packed_dim_info_set_empty(self):
        # Empty other has no PDIs to check, so any self is compatible
        self.assertTrue(
            CHANNELS_PACKED_TEXTURE.has_compatible_packed_dim_info_set(NO_STORAGE)
        )
        self.assertTrue(NO_STORAGE.has_compatible_packed_dim_info_set(NO_STORAGE))

    def test_has_compatible_packed_dim_info_set_buffer_and_texture(self):
        # CHANNELS_PACKED_ANY has both buffer and texture PDIs with packed_dim=2
        # ANY_STORAGE is a superset
        self.assertTrue(
            ANY_STORAGE.has_compatible_packed_dim_info_set(CHANNELS_PACKED_ANY)
        )
        # CHANNELS_PACKED_TEXTURE only has texture PDIs, not buffer
        self.assertFalse(
            CHANNELS_PACKED_TEXTURE.has_compatible_packed_dim_info_set(
                CHANNELS_PACKED_ANY
            )
        )

    def test_has_compatible_packed_dim_info_set_quantized(self):
        # PACKED_INT8_4W4C and PACKED_INT8_4C1W both produce PackedDimInfo(2, 4)
        self.assertTrue(
            PACKED_INT8_4W4C_BUFFER.has_compatible_packed_dim_info_set(
                PACKED_INT8_4C1W_BUFFER
            )
        )
        # PACKED_INT8_BUFFER has all three quantized layouts (packed_dim 0 and 2)
        # so a single packed_dim=2 layout is not a superset
        self.assertFalse(
            PACKED_INT8_4W4C_BUFFER.has_compatible_packed_dim_info_set(
                PACKED_INT8_BUFFER
            )
        )

    # -- constrain_to_compatible_packed_dim --

    def test_constrain_to_compatible_packed_dim(self):
        full = ANY_TEXTURE
        constraint = CHANNELS_PACKED_TEXTURE
        result = full.constrain_to_compatible_packed_dim(constraint)
        # Only channels-packed layouts have packed dim 2
        self.assertIn(
            VkMemoryLayout.TENSOR_CHANNELS_PACKED, result.valid_texture_layouts
        )
        self.assertNotIn(
            VkMemoryLayout.TENSOR_WIDTH_PACKED, result.valid_texture_layouts
        )
        self.assertNotIn(
            VkMemoryLayout.TENSOR_HEIGHT_PACKED, result.valid_texture_layouts
        )

    def test_constrain_to_compatible_packed_dim_empty_other(self):
        full = ANY_TEXTURE
        result = full.constrain_to_compatible_packed_dim(NO_STORAGE)
        self.assertEqual(result, full)

    def test_constrain_to_compatible_packed_dim_buffer(self):
        result = ANY_BUFFER.constrain_to_compatible_packed_dim(CONTIGUOUS_BUFFER)
        # CONTIGUOUS_BUFFER is width-packed → PackedDimInfo(0, 1)
        # Only TENSOR_WIDTH_PACKED has the same PDI among non-quantized layouts
        self.assertIn(VkMemoryLayout.TENSOR_WIDTH_PACKED, result.valid_buffer_layouts)
        self.assertNotIn(
            VkMemoryLayout.TENSOR_CHANNELS_PACKED, result.valid_buffer_layouts
        )
        self.assertNotIn(
            VkMemoryLayout.TENSOR_HEIGHT_PACKED, result.valid_buffer_layouts
        )

    def test_constrain_to_compatible_packed_dim_both_storage_types(self):
        result = ANY_STORAGE.constrain_to_compatible_packed_dim(CHANNELS_PACKED_ANY)
        # Should keep only channels-packed layouts in both buffer and texture
        self.assertIn(
            VkMemoryLayout.TENSOR_CHANNELS_PACKED, result.valid_buffer_layouts
        )
        self.assertIn(
            VkMemoryLayout.TENSOR_CHANNELS_PACKED, result.valid_texture_layouts
        )
        self.assertNotIn(
            VkMemoryLayout.TENSOR_WIDTH_PACKED, result.valid_buffer_layouts
        )
        self.assertNotIn(
            VkMemoryLayout.TENSOR_WIDTH_PACKED, result.valid_texture_layouts
        )

    def test_constrain_to_compatible_packed_dim_disjoint(self):
        # Width-packed and channels-packed have different packed dims
        result = WIDTH_PACKED_TEXTURE.constrain_to_compatible_packed_dim(
            CHANNELS_PACKED_TEXTURE
        )
        self.assertTrue(result.is_empty())

    def test_constrain_to_compatible_packed_dim_is_independent_copy(self):
        original = ANY_TEXTURE.copy()
        result = ANY_TEXTURE.constrain_to_compatible_packed_dim(CHANNELS_PACKED_TEXTURE)
        # Original should not be modified
        self.assertEqual(ANY_TEXTURE, original)
        self.assertNotEqual(result, ANY_TEXTURE)

    # -- Convenience constants --

    def test_convenience_constants(self):
        self.assertFalse(CONTIGUOUS_ANY.is_empty())
        self.assertFalse(CONTIGUOUS_BUFFER.is_empty())
        self.assertFalse(WIDTH_PACKED_TEXTURE.is_empty())
        self.assertFalse(HEIGHT_PACKED_TEXTURE.is_empty())
        self.assertFalse(CHANNELS_PACKED_TEXTURE.is_empty())
        self.assertFalse(CHANNELS_PACKED_ANY.is_empty())
        self.assertFalse(ANY_TEXTURE.is_empty())
        self.assertFalse(ANY_BUFFER.is_empty())
        self.assertFalse(ANY_STORAGE.is_empty())
        self.assertTrue(NO_STORAGE.is_empty())

    # -- make_tensor_repset --

    def test_make_tensor_repset_buffer(self):
        tr = TensorRepr(VkStorageType.BUFFER, VkMemoryLayout.TENSOR_WIDTH_PACKED)
        repset = make_tensor_repset(tr)
        self.assertEqual(
            repset.valid_buffer_layouts, {VkMemoryLayout.TENSOR_WIDTH_PACKED}
        )
        self.assertEqual(repset.valid_texture_layouts, set())

    def test_make_tensor_repset_texture(self):
        tr = TensorRepr(VkStorageType.TEXTURE_3D, VkMemoryLayout.TENSOR_CHANNELS_PACKED)
        repset = make_tensor_repset(tr)
        self.assertEqual(repset.valid_buffer_layouts, set())
        self.assertEqual(
            repset.valid_texture_layouts, {VkMemoryLayout.TENSOR_CHANNELS_PACKED}
        )


class TestTensorRepSetList(unittest.TestCase):
    def test_single_element_broadcasting(self):
        repset = CHANNELS_PACKED_TEXTURE
        lst = TensorRepSetList(repset)
        self.assertEqual(len(lst), 1)
        # Accessing index > 0 broadcasts to the single element
        self.assertEqual(lst[0], repset)
        self.assertEqual(lst[2], repset)

    def test_multi_element_indexing(self):
        a = CHANNELS_PACKED_TEXTURE
        b = WIDTH_PACKED_TEXTURE
        lst = TensorRepSetList([a, b])
        self.assertEqual(len(lst), 2)
        self.assertEqual(lst[0], a)
        self.assertEqual(lst[1], b)

    def test_setitem_single(self):
        lst = TensorRepSetList(CHANNELS_PACKED_TEXTURE)
        lst[0] = WIDTH_PACKED_TEXTURE
        self.assertEqual(lst[0], WIDTH_PACKED_TEXTURE)

    def test_setitem_single_broadcast(self):
        lst = TensorRepSetList(CHANNELS_PACKED_TEXTURE)
        # Setting index > 0 on a single-element list updates the single element
        lst[3] = WIDTH_PACKED_TEXTURE
        self.assertEqual(lst[0], WIDTH_PACKED_TEXTURE)

    def test_setitem_multi(self):
        lst = TensorRepSetList([CHANNELS_PACKED_TEXTURE, WIDTH_PACKED_TEXTURE])
        lst[1] = HEIGHT_PACKED_TEXTURE
        self.assertEqual(lst[1], HEIGHT_PACKED_TEXTURE)
        self.assertEqual(lst[0], CHANNELS_PACKED_TEXTURE)

    def test_append(self):
        lst = TensorRepSetList([])
        lst.append(CHANNELS_PACKED_TEXTURE)
        lst.append(WIDTH_PACKED_TEXTURE)
        self.assertEqual(len(lst), 2)

    def test_any_is_empty_true(self):
        lst = TensorRepSetList([CHANNELS_PACKED_TEXTURE, NO_STORAGE])
        self.assertTrue(lst.any_is_empty())

    def test_any_is_empty_false(self):
        lst = TensorRepSetList([CHANNELS_PACKED_TEXTURE, WIDTH_PACKED_TEXTURE])
        self.assertFalse(lst.any_is_empty())

    def test_any_is_empty_no_elements(self):
        lst = TensorRepSetList([])
        self.assertTrue(lst.any_is_empty())

    def test_str(self):
        lst = TensorRepSetList([CHANNELS_PACKED_TEXTURE])
        s = str(lst)
        self.assertIn("TensorRepSet", s)


class TestOpRepSets(unittest.TestCase):
    """
    Tests for OpRepSets using mock torch.fx.Node objects. The constructor
    requires a node with .op, .target, .args, and .meta["val"] attributes.
    """

    def _make_unary_op(self, input_shape=(1, 3, 8, 8), repset=ANY_STORAGE):
        """Create an OpRepSets for a simple unary op (single tensor in, single tensor out)."""
        arg = _make_tensor_arg_node(input_shape)
        out_val = _make_fake_tensor(input_shape)
        node = _make_op_node(
            target=torch.ops.aten.relu.default,
            args=(arg,),
            output_val=out_val,
        )
        return OpRepSets(
            TensorRepSetList(repset),
            TensorRepSetList(repset),
            node,
            DEFAULT_TEXTURE_LIMITS,
        )

    def _make_binary_op(
        self,
        shape_a=(1, 3, 8, 8),
        shape_b=(1, 3, 8, 8),
        repset=ANY_STORAGE,
    ):
        """Create an OpRepSets for a binary op (two tensor inputs, single tensor output)."""
        arg_a = _make_tensor_arg_node(shape_a)
        arg_b = _make_tensor_arg_node(shape_b)
        out_val = _make_fake_tensor(shape_a)
        node = _make_op_node(
            target=torch.ops.aten.add.Tensor,
            args=(arg_a, arg_b),
            output_val=out_val,
        )
        return OpRepSets(
            TensorRepSetList(repset),
            TensorRepSetList(repset),
            node,
            DEFAULT_TEXTURE_LIMITS,
        )

    # -- Construction --

    def test_unary_op_construction(self):
        op_repsets = self._make_unary_op()
        self.assertFalse(op_repsets.any_is_empty())
        self.assertEqual(op_repsets.primary_arg_idx, 0)
        self.assertTrue(op_repsets.sync_primary_io_repr)

    def test_binary_op_syncs_args(self):
        """When a single repset covers all inputs, sync_args_repr is True."""
        op_repsets = self._make_binary_op()
        self.assertTrue(op_repsets.sync_args_repr)
        self.assertEqual(op_repsets.primary_arg_idx, 0)

    def test_binary_op_separate_repsets_no_sync(self):
        """When each input has its own repset, sync_args_repr is False."""
        arg_a = _make_tensor_arg_node((1, 3, 8, 8))
        arg_b = _make_tensor_arg_node((1, 3, 8, 8))
        out_val = _make_fake_tensor((1, 3, 8, 8))
        node = _make_op_node(
            target=torch.ops.aten.add.Tensor,
            args=(arg_a, arg_b),
            output_val=out_val,
        )
        op_repsets = OpRepSets(
            TensorRepSetList([CHANNELS_PACKED_ANY, WIDTH_PACKED_TEXTURE]),
            TensorRepSetList(ANY_STORAGE),
            node,
            DEFAULT_TEXTURE_LIMITS,
        )
        self.assertFalse(op_repsets.sync_args_repr)

    def test_no_sync_primary_io_when_different_repsets(self):
        """sync_primary_io_repr is False when input and output repsets differ."""
        arg = _make_tensor_arg_node((1, 3, 8, 8))
        out_val = _make_fake_tensor((1, 3, 8, 8))
        node = _make_op_node(
            target=torch.ops.aten.relu.default,
            args=(arg,),
            output_val=out_val,
        )
        op_repsets = OpRepSets(
            TensorRepSetList(CHANNELS_PACKED_ANY),
            TensorRepSetList(WIDTH_PACKED_TEXTURE),
            node,
            DEFAULT_TEXTURE_LIMITS,
        )
        self.assertFalse(op_repsets.sync_primary_io_repr)

    # -- Scalar args are skipped --

    def test_scalar_arg_skipped(self):
        """Non-tensor args should be treated as ALL_STORAGES_REPSET."""
        tensor_arg = _make_tensor_arg_node((1, 3, 8, 8))
        # Second arg is a scalar (float)
        scalar_arg = 1.0
        out_val = _make_fake_tensor((1, 3, 8, 8))
        node = _make_op_node(
            target=torch.ops.aten.add.Tensor,
            args=(tensor_arg, scalar_arg),
            output_val=out_val,
        )
        op_repsets = OpRepSets(
            TensorRepSetList(ANY_STORAGE),
            TensorRepSetList(ANY_STORAGE),
            node,
            DEFAULT_TEXTURE_LIMITS,
        )
        self.assertFalse(op_repsets.any_is_empty())
        # The scalar arg should get ALL_STORAGES_REPSET
        # self.assertEqual(op_repsets.get_arg_repset(1), ALL_STORAGES_REPSET, f"""{op_repsets.get_arg_repset(1)}""")

    # -- pick_representations --

    def test_pick_representations_unary(self):
        op_repsets = self._make_unary_op(repset=CHANNELS_PACKED_TEXTURE)
        args_repr, outs_repr = op_repsets.pick_representations()
        self.assertEqual(len(args_repr), 1)
        self.assertEqual(len(outs_repr), 1)
        self.assertEqual(args_repr[0].storage_type, VkStorageType.TEXTURE_3D)
        self.assertEqual(
            args_repr[0].memory_layout, VkMemoryLayout.TENSOR_CHANNELS_PACKED
        )
        self.assertEqual(outs_repr[0].storage_type, VkStorageType.TEXTURE_3D)
        self.assertEqual(
            outs_repr[0].memory_layout, VkMemoryLayout.TENSOR_CHANNELS_PACKED
        )

    def test_pick_representations_prefers_texture(self):
        op_repsets = self._make_unary_op(repset=ANY_STORAGE)
        _, outs_repr = op_repsets.pick_representations()
        self.assertEqual(outs_repr[0].storage_type, VkStorageType.TEXTURE_3D)

    def test_pick_representations_buffer_only(self):
        op_repsets = self._make_unary_op(repset=CONTIGUOUS_BUFFER)
        args_repr, outs_repr = op_repsets.pick_representations()
        self.assertEqual(args_repr[0].storage_type, VkStorageType.BUFFER)
        self.assertEqual(outs_repr[0].storage_type, VkStorageType.BUFFER)

    # -- try_constrain_with_arg_repset --

    def test_try_constrain_with_arg_repset_narrows(self):
        op_repsets = self._make_unary_op(repset=ANY_STORAGE)
        changed = op_repsets.try_constrain_with_arg_repset(0, CHANNELS_PACKED_TEXTURE)
        self.assertTrue(changed)
        arg_repset = op_repsets.get_arg_repset(0)
        self.assertTrue(arg_repset.texture_is_valid())
        # After constraining to channels-packed texture, only channels-packed
        # layouts should remain
        self.assertIn(
            VkMemoryLayout.TENSOR_CHANNELS_PACKED, arg_repset.valid_texture_layouts
        )

    def test_try_constrain_with_arg_repset_no_common(self):
        """Returns False when source repset has nothing in common."""
        op_repsets = self._make_unary_op(repset=CHANNELS_PACKED_TEXTURE)
        changed = op_repsets.try_constrain_with_arg_repset(0, CONTIGUOUS_BUFFER)
        self.assertFalse(changed)

    def test_try_constrain_with_arg_repset_same_repset(self):
        """Returns False when source repset equals current repset."""
        op_repsets = self._make_unary_op(repset=CHANNELS_PACKED_TEXTURE)
        changed = op_repsets.try_constrain_with_arg_repset(0, CHANNELS_PACKED_TEXTURE)
        self.assertFalse(changed)

    def test_try_constrain_propagates_to_synced_args(self):
        """When sync_args_repr is True, constraining one arg propagates to the other."""
        op_repsets = self._make_binary_op(repset=ANY_STORAGE)
        op_repsets.try_constrain_with_arg_repset(0, CHANNELS_PACKED_TEXTURE)
        arg0 = op_repsets.get_arg_repset(0)
        arg1 = op_repsets.get_arg_repset(1)
        # arg1 should also be constrained to have a compatible packed dim
        self.assertTrue(arg0.has_compatible_packed_dim_info_set(arg1))

    def test_try_constrain_propagates_to_output(self):
        """When sync_primary_io_repr is True, constraining the primary arg also
        constrains the output."""
        op_repsets = self._make_unary_op(repset=ANY_STORAGE)
        op_repsets.try_constrain_with_arg_repset(0, CHANNELS_PACKED_TEXTURE)
        out_repset = op_repsets.get_out_repset(0)
        arg_repset = op_repsets.get_arg_repset(0)
        self.assertTrue(out_repset.has_compatible_packed_dim_info_set(arg_repset))

    # -- try_constrain_with_out_repset --

    def test_try_constrain_with_out_repset_when_io_not_synced(self):
        """Output can be constrained independently when sync_primary_io_repr is False."""
        arg = _make_tensor_arg_node((1, 3, 8, 8))
        out_val = _make_fake_tensor((1, 3, 8, 8))
        node = _make_op_node(
            target=torch.ops.aten.relu.default,
            args=(arg,),
            output_val=out_val,
        )
        op_repsets = OpRepSets(
            TensorRepSetList(CHANNELS_PACKED_TEXTURE),
            TensorRepSetList(ANY_STORAGE),
            node,
            DEFAULT_TEXTURE_LIMITS,
        )
        self.assertFalse(op_repsets.sync_primary_io_repr)
        changed = op_repsets.try_constrain_with_out_repset(WIDTH_PACKED_TEXTURE)
        self.assertTrue(changed)
        out = op_repsets.get_out_repset(0)
        self.assertIn(VkMemoryLayout.TENSOR_WIDTH_PACKED, out.valid_texture_layouts)

    def test_try_constrain_with_out_repset_skipped_when_synced(self):
        """try_constrain_with_out_repset narrows the output even when sync_primary_io_repr is True."""
        op_repsets = self._make_unary_op(repset=ANY_STORAGE)
        self.assertTrue(op_repsets.sync_primary_io_repr)
        changed = op_repsets.try_constrain_with_out_repset(CHANNELS_PACKED_TEXTURE)
        self.assertTrue(changed)
        out = op_repsets.get_out_repset(0)
        self.assertIn(VkMemoryLayout.TENSOR_CHANNELS_PACKED, out.valid_texture_layouts)

    # -- Multiple output tensors --

    def test_multiple_outputs_no_sync(self):
        """When each output has its own repset, sync_outs_repr is False."""
        arg = _make_tensor_arg_node((1, 3, 8, 8))
        out0 = _make_fake_tensor((1, 3, 8, 8))
        out1 = _make_fake_tensor((1, 3, 8, 8))
        node = _make_op_node(
            target=torch.ops.aten.relu.default,
            args=(arg,),
            output_val=[out0, out1],
        )
        op_repsets = OpRepSets(
            TensorRepSetList(ANY_STORAGE),
            TensorRepSetList([ANY_STORAGE, CHANNELS_PACKED_ANY]),
            node,
            DEFAULT_TEXTURE_LIMITS,
        )
        self.assertFalse(op_repsets.sync_outs_repr)
        self.assertFalse(op_repsets.any_is_empty())

    # -- High dimensional tensors --

    def test_high_dim_tensor_filters_texture_layouts(self):
        """Tensors with >4 dims should have texture layouts filtered out."""
        shape = (2, 3, 4, 5, 6)  # 5 dimensions
        op_repsets = self._make_unary_op(input_shape=shape, repset=ANY_STORAGE)
        # The arg repset should have no valid texture layouts for high-dim tensors
        arg_repset = op_repsets.get_arg_repset(0)
        self.assertFalse(arg_repset.texture_is_valid())
        self.assertTrue(arg_repset.buffer_is_valid())

    # -- getitem operator --

    def test_getitem_op(self):
        """OpRepSets should handle operator.getitem correctly."""
        # Create a node that produces a tuple of tensors
        parent_arg = _make_tensor_arg_node((1, 3, 8, 8))
        parent_fake_0 = _make_fake_tensor((1, 3, 8, 8))
        parent_fake_1 = _make_fake_tensor((1, 3, 8, 8))
        parent_arg.meta = {"val": [parent_fake_0, parent_fake_1]}

        out_val = _make_fake_tensor((1, 3, 8, 8))
        node = _make_op_node(
            target=operator.getitem,
            args=(parent_arg, 0),
            output_val=out_val,
        )
        op_repsets = OpRepSets(
            TensorRepSetList(ANY_STORAGE),
            TensorRepSetList(ANY_STORAGE),
            node,
            DEFAULT_TEXTURE_LIMITS,
        )
        self.assertFalse(op_repsets.any_is_empty())

    # -- Quantized binary ops with different layouts but same packed dim --

    def _make_quantized_binary_op(
        self,
        args_repset,
        outs_repset,
        shape_a=(1, 3, 8, 8),
        shape_b=(1, 3, 8, 8),
    ):
        """Create an OpRepSets for a quantized binary op with separate arg/out repsets."""
        arg_a = _make_tensor_arg_node(shape_a)
        arg_b = _make_tensor_arg_node(shape_b)
        out_val = _make_fake_tensor(shape_a)
        node = _make_op_node(
            target=torch.ops.aten.add.Tensor,
            args=(arg_a, arg_b),
            output_val=out_val,
        )
        return OpRepSets(
            TensorRepSetList(args_repset),
            TensorRepSetList(outs_repset),
            node,
            DEFAULT_TEXTURE_LIMITS,
        )

    def test_quantized_binary_different_layouts_same_packed_dim(self):
        """Args and output can have different quantized layouts if packed dim matches."""
        # PACKED_INT8_4W4C and PACKED_INT8_4C1W both have packed_dim=2
        op_repsets = self._make_quantized_binary_op(
            args_repset=PACKED_INT8_4W4C_BUFFER,
            outs_repset=PACKED_INT8_4C1W_BUFFER,
        )
        self.assertFalse(op_repsets.sync_primary_io_repr)
        self.assertFalse(op_repsets.any_is_empty())

        arg0 = op_repsets.get_arg_repset(0)
        out = op_repsets.get_out_repset(0)
        self.assertIn(VkMemoryLayout.PACKED_INT8_4W4C, arg0.valid_buffer_layouts)
        self.assertIn(VkMemoryLayout.PACKED_INT8_4C1W, out.valid_buffer_layouts)

    def test_quantized_binary_constrain_arg_with_synced_io(self):
        """When args and output share the same repset (sync_primary_io_repr=True),
        constraining an arg to a specific quantized layout also narrows the output
        to layouts with a compatible packed dim."""
        op_repsets = self._make_quantized_binary_op(
            args_repset=PACKED_INT8_CHANNELS_PACKED_BUFFER,
            outs_repset=PACKED_INT8_CHANNELS_PACKED_BUFFER,
        )
        self.assertTrue(op_repsets.sync_primary_io_repr)
        changed = op_repsets.try_constrain_with_arg_repset(0, PACKED_INT8_4W4C_BUFFER)
        self.assertTrue(changed)
        arg0 = op_repsets.get_arg_repset(0)
        self.assertIn(VkMemoryLayout.PACKED_INT8_4W4C, arg0.valid_buffer_layouts)
        self.assertNotIn(VkMemoryLayout.PACKED_INT8_4C1W, arg0.valid_buffer_layouts)
        # Output should be narrowed to compatible packed dim layouts
        out = op_repsets.get_out_repset(0)
        self.assertTrue(out.has_compatible_packed_dim_info_set(arg0))

    def test_quantized_binary_synced_args_different_out(self):
        """Synced args can be constrained together while output uses a different
        quantized layout with the same packed dim."""
        # Use shared repset for args so sync_args_repr=True
        op_repsets = self._make_quantized_binary_op(
            args_repset=PACKED_INT8_BUFFER,
            outs_repset=PACKED_INT8_BUFFER,
        )
        self.assertTrue(op_repsets.sync_args_repr)
        changed = op_repsets.try_constrain_with_arg_repset(0, PACKED_INT8_4W4C_BUFFER)
        self.assertTrue(changed)
        arg0 = op_repsets.get_arg_repset(0)
        arg1 = op_repsets.get_arg_repset(1)
        # arg0 is narrowed to PACKED_INT8_4W4C
        self.assertIn(VkMemoryLayout.PACKED_INT8_4W4C, arg0.valid_buffer_layouts)
        # arg1 should be constrained to layouts with compatible packed dim (=2)
        self.assertTrue(arg1.has_compatible_packed_dim_info_set(arg0))

    def test_quantized_binary_constrain_out_with_compatible_packed_dim(self):
        """Output can be constrained to a different quantized layout as long as
        packed dim is compatible."""
        op_repsets = self._make_quantized_binary_op(
            args_repset=PACKED_INT8_CHANNELS_PACKED_BUFFER,
            outs_repset=PACKED_INT8_CHANNELS_PACKED_BUFFER,
        )
        changed = op_repsets.try_constrain_with_out_repset(PACKED_INT8_4C1W_BUFFER)
        self.assertTrue(changed)
        out = op_repsets.get_out_repset(0)
        self.assertIn(VkMemoryLayout.PACKED_INT8_4C1W, out.valid_buffer_layouts)
        self.assertNotIn(VkMemoryLayout.PACKED_INT8_4W4C, out.valid_buffer_layouts)

    def test_quantized_binary_incompatible_packed_dim_no_common(self):
        """Args and output with different packed dims have nothing in common."""
        # PACKED_INT8_4W4C has packed_dim=2, PACKED_INT8_4W has packed_dim=0
        op_repsets = self._make_quantized_binary_op(
            args_repset=PACKED_INT8_4W4C_BUFFER,
            outs_repset=PACKED_INT8_4W_BUFFER,
        )
        self.assertFalse(op_repsets.sync_primary_io_repr)
        # Constraining arg to width-packed should fail since arg is channels-packed
        changed = op_repsets.try_constrain_with_arg_repset(0, PACKED_INT8_4W_BUFFER)
        self.assertFalse(changed)


class TestTensorReprList(unittest.TestCase):
    def test_single_element_broadcasting(self):
        tr = TensorRepr(VkStorageType.TEXTURE_3D, VkMemoryLayout.TENSOR_CHANNELS_PACKED)
        lst = TensorReprList(tr)
        self.assertEqual(len(lst), 1)
        self.assertEqual(lst[0], tr)
        self.assertEqual(lst[5], tr)

    def test_multi_element(self):
        a = TensorRepr(VkStorageType.TEXTURE_3D, VkMemoryLayout.TENSOR_CHANNELS_PACKED)
        b = TensorRepr(VkStorageType.BUFFER, VkMemoryLayout.TENSOR_WIDTH_PACKED)
        lst = TensorReprList([a, b])
        self.assertEqual(len(lst), 2)
        self.assertEqual(lst[0], a)
        self.assertEqual(lst[1], b)

    def test_setitem(self):
        a = TensorRepr(VkStorageType.TEXTURE_3D, VkMemoryLayout.TENSOR_CHANNELS_PACKED)
        b = TensorRepr(VkStorageType.BUFFER, VkMemoryLayout.TENSOR_WIDTH_PACKED)
        lst = TensorReprList([a, b])
        c = TensorRepr(VkStorageType.TEXTURE_3D, VkMemoryLayout.TENSOR_WIDTH_PACKED)
        lst[1] = c
        self.assertEqual(lst[1], c)

    def test_append(self):
        lst = TensorReprList([])
        tr = TensorRepr(VkStorageType.TEXTURE_3D, VkMemoryLayout.TENSOR_CHANNELS_PACKED)
        lst.append(tr)
        self.assertEqual(len(lst), 1)
        self.assertEqual(lst[0], tr)

    def test_storage_type_and_memory_layout(self):
        tr = TensorRepr(VkStorageType.TEXTURE_3D, VkMemoryLayout.TENSOR_CHANNELS_PACKED)
        lst = TensorReprList(tr)
        self.assertEqual(lst.storage_type(), VkStorageType.TEXTURE_3D)
        self.assertEqual(lst.memory_layout(), VkMemoryLayout.TENSOR_CHANNELS_PACKED)

    def test_equality(self):
        a = TensorRepr(VkStorageType.TEXTURE_3D, VkMemoryLayout.TENSOR_CHANNELS_PACKED)
        lst1 = TensorReprList(a)
        lst2 = TensorReprList(a)
        self.assertEqual(lst1, lst2)

    def test_inequality_different_length(self):
        a = TensorRepr(VkStorageType.TEXTURE_3D, VkMemoryLayout.TENSOR_CHANNELS_PACKED)
        b = TensorRepr(VkStorageType.BUFFER, VkMemoryLayout.TENSOR_WIDTH_PACKED)
        lst1 = TensorReprList(a)
        lst2 = TensorReprList([a, b])
        self.assertNotEqual(lst1, lst2)

    def test_str(self):
        tr = TensorRepr(VkStorageType.TEXTURE_3D, VkMemoryLayout.TENSOR_CHANNELS_PACKED)
        lst = TensorReprList(tr)
        s = str(lst)
        self.assertIn("TensorRepr", s)


if __name__ == "__main__":
    unittest.main()
