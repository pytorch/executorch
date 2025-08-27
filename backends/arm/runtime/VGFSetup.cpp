/*
 * Copyright 2025 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * VGF functions which prepare a graph for execution by allocating the
 * appropriate vulkan structures.
 */

#include <executorch/backends/arm/runtime/VGFSetup.h>

#include <vgf/decoder.hpp>
#include <vgf/vulkan_helpers.generated.hpp>

using namespace mlsdk;

namespace executorch {
namespace backends {
namespace vgf {

/* static function to map format to byte count */
static uint32_t get_format_size(VkFormat format);

// Debug function to inspect memory properties
static string memory_flags_to_string(VkMemoryPropertyFlags flags) {
  if (flags == 0)
    return "0";

  vector<string> parts;
#define TRY_FLAG(f)         \
  if (flags & (f)) {        \
    parts.emplace_back(#f); \
    flags &= ~(f);          \
  }

  TRY_FLAG(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  TRY_FLAG(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
  TRY_FLAG(VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
  TRY_FLAG(VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
  TRY_FLAG(VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)
#ifdef VK_MEMORY_PROPERTY_PROTECTED_BIT
  TRY_FLAG(VK_MEMORY_PROPERTY_PROTECTED_BIT)
#endif
#undef TRY_FLAG

  if (flags) {
    // any leftover bits we didn’t name
    ostringstream hex;
    hex << "0x" << std::hex << flags;
    parts.emplace_back(hex.str());
  }

  ostringstream joined;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i)
      joined << " | ";
    joined << parts[i];
  }
  return joined.str();
}

/**
 * Tensor free helper function
 */
void free_tensor(
    VkDevice device,
    VkTensorViewARM tensor_view,
    VkTensorARM tensor,
    VkDeviceMemory memory) {
  vkDestroyTensorViewARM(device, tensor_view, nullptr);
  vkDestroyTensorARM(device, tensor, nullptr);
  vkFreeMemory(device, memory, nullptr);
}

uint32_t get_memory_index(
    VkPhysicalDevice vk_physical,
    VkMemoryRequirements2 memory_requirements,
    VkMemoryPropertyFlags aims) {
  VkPhysicalDeviceMemoryProperties mem_properties;
  vkGetPhysicalDeviceMemoryProperties(vk_physical, &mem_properties);

  uint32_t memory_type = 0;
  for (size_t i = 0; i < 31; ++i) {
    if (memory_requirements.memoryRequirements.memoryTypeBits & (0x1 << i)) {
      memory_type = i;
      if ((mem_properties.memoryTypes[i].propertyFlags & aims) == aims)
        break;
    }
  }
  return memory_type;
}

/**
 * Tensor allocation helper function
 */
VkResult allocate_tensor(
    VkPhysicalDevice physical,
    VkDevice device,
    VkFormat format,
    uint32_t shape_size,
    const int64_t* shape,
    uint32_t stride_size,
    const int64_t* stride,
    VkTensorDescriptionARM* description,
    VkTensorViewARM* tensor_view,
    VkTensorARM* tensor,
    VkDeviceMemory* memory) {
  VkResult result;

  *description = VkTensorDescriptionARM{
      .sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
      .pNext = nullptr,
      .tiling = VK_TENSOR_TILING_LINEAR_ARM,
      .format = format,
      .dimensionCount = shape_size,
      .pDimensions = shape,
      // Note: stride_data of 0's causes size==0, null means stride==size
      .pStrides = (0 == stride_size ? nullptr : stride),
      .usage = VK_TENSOR_USAGE_SHADER_BIT_ARM |
          VK_TENSOR_USAGE_TRANSFER_SRC_BIT_ARM |
          VK_TENSOR_USAGE_TRANSFER_DST_BIT_ARM |
          VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM,
  };
  const VkTensorCreateInfoARM create_info = {
      .sType = VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM,
      .pNext = nullptr,
      .flags = 0,
      .pDescription = description,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
  };

  result = vkCreateTensorARM(device, &create_info, nullptr, tensor);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to CreateTensor, error %d", result);
    return result;
  }

  // Get backing memory requirements
  const VkTensorMemoryRequirementsInfoARM memory_requirements_info = {
      .sType = VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_ARM,
      .pNext = nullptr,
      .tensor = *tensor,
  };
  VkMemoryRequirements2 memory_requirements = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
      .pNext = nullptr,
  };
  vkGetTensorMemoryRequirementsARM(
      device, &memory_requirements_info, &memory_requirements);

  VkMemoryPropertyFlags aims = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  uint32_t memory_index = get_memory_index(physical, memory_requirements, aims);

  // Allocate memory
  const VkMemoryAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = nullptr,
      .allocationSize = memory_requirements.memoryRequirements.size,
      .memoryTypeIndex = memory_index,
  };

  vkAllocateMemory(device, &allocate_info, nullptr, memory);

  // Bind tensor to memory
  const VkBindTensorMemoryInfoARM bind_info = {
      .sType = VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_ARM,
      .pNext = nullptr,
      .tensor = *tensor,
      .memory = *memory,
      .memoryOffset = 0,
  };
  vkBindTensorMemoryARM(device, 1, &bind_info);

  VkTensorViewCreateInfoARM tensor_view_info = {
      .sType = VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_ARM,
      .pNext = nullptr,
      .flags = 0,
      .tensor = *tensor,
      .format = format,
  };
  VkResult res_tv =
      vkCreateTensorViewARM(device, &tensor_view_info, nullptr, tensor_view);
  ET_LOG(Info, "    tensor view (success %d)", res_tv == VK_SUCCESS);

  return res_tv;
}

static void debug_print_sequence(
    unique_ptr<vgflib::ModelSequenceTableDecoder>& sequence_decoder) {
  ET_LOG(Info, "VGF Sequences:");
  for (int i = 0; i < sequence_decoder->modelSequenceTableSize(); i++) {
    ET_LOG(
        Info,
        "  Sequence(%d) '%s':",
        i,
        string(sequence_decoder->getSegmentName(i)).c_str());
    auto dispatch_shape = sequence_decoder->getSegmentDispatchShape(i);
    ET_LOG(
        Info,
        "    dispatch shape %d %d %d",
        dispatch_shape[0],
        dispatch_shape[1],
        dispatch_shape[2]);
    ET_LOG(
        Info,
        "    is graph? %d",
        vgflib::ModuleType::GRAPH == sequence_decoder->getSegmentType(i));
    ET_LOG(
        Info,
        "    module index %d",
        sequence_decoder->getSegmentModuleIndex(i));
    auto input_names = sequence_decoder->getModelSequenceInputNamesHandle();
    ET_LOG(
        Info, "    names (%ld):", sequence_decoder->getNamesSize(input_names));
    for (int j = 0; j < sequence_decoder->getNamesSize(input_names); j++) {
      ET_LOG(
          Info,
          "      %d: %s",
          j,
          string(sequence_decoder->getName(input_names, j)).c_str());
    }
  }
}

static void debug_print_resources(
    unique_ptr<vgflib::ModelResourceTableDecoder>& resource_decoder) {
  ET_LOG(Info, "Resources:");
  for (int i = 0; i < resource_decoder->size(); i++) {
    ET_LOG(Info, "  MRT entry %d", i);
    if (!resource_decoder->getDescriptorType(i).has_value()) {
      ET_LOG(Info, "    DescriptorType NONE");
    } else {
      ET_LOG(
          Info,
          "    DescriptorType %u, is tensor? %d",
          resource_decoder->getDescriptorType(i).value(),
          resource_decoder->getDescriptorType(i).value() ==
              VK_DESCRIPTOR_TYPE_TENSOR_ARM);
    }
    ET_LOG(
        Info,
        "    VkFormat %u from vgf format %u",
        vgflib::ToVkFormat(resource_decoder->getVkFormat(i)),
        resource_decoder->getVkFormat(i));
    switch (resource_decoder->getCategory(i)) {
      case vgflib::ResourceCategory::INPUT:
      case vgflib::ResourceCategory::OUTPUT: {
        ET_LOG(Info, "    Category INPUT/OUTPUT");
        // Get tensor shape and strides
        auto shape = resource_decoder->getTensorShape(i);
        const vector<int64_t> the_shape(shape.begin(), shape.end());
        auto stride = resource_decoder->getTensorStride(i);
        const vector<int64_t> the_stride(stride.begin(), stride.end());
        ET_LOG(
            Info,
            "    rank %ld, stride rank %ld",
            the_shape.size(),
            the_stride.size());
        for (int j = 0; j < the_shape.size(); j++) {
          ET_LOG(Info, "      %d: dim %ld", j, the_shape[j]);
        }
        // Allocate a tensor with bound memory
        break;
      }
      case vgflib::ResourceCategory::INTERMEDIATE:
        ET_LOG(Info, "    Category INTERMEDIATE");
        break;
      case vgflib::ResourceCategory::CONSTANT:
        ET_LOG(Info, "    Category CONSTANT");
        break;
      default:
        ET_LOG(Info, "    Category UNKNOWN");
        break;
    }
  }
}

static void debug_print_modules(
    unique_ptr<vgflib::ModuleTableDecoder>& module_decoder) {
  ET_LOG(Info, "VGF Modules:");
  for (int i = 0; i < module_decoder->size(); i++) {
    auto name = string(module_decoder->getModuleName(i));
    auto entrypoint = string(module_decoder->getModuleEntryPoint(i));
    auto type = module_decoder->getModuleType(i);
    auto spirv = module_decoder->getModuleCode(i);
    ET_LOG(Info, "  Module(%d) '%s':", i, name.c_str());
    ET_LOG(
        Info,
        "    is graph? %d",
        vgflib::ModuleType::GRAPH == module_decoder->getModuleType(i));
    ET_LOG(Info, "    entrypoint '%s'", entrypoint.c_str());
    ET_LOG(Info, "    has spirv %d", module_decoder->hasSPIRV(i));
    ET_LOG(
        Info, "    code size %lu", spirv.size()); // read the .begin() to .end()
  }
}

bool VgfRepr::process_vgf(const char* vgf_data, ArrayRef<CompileSpec> specs) {
  ET_LOG(Info, "Preparing VGF as Vulkan objects");

  VkResult result;

  // Prepare temporary decoders
  unique_ptr<vgflib::HeaderDecoder> header_decoder =
      vgflib::CreateHeaderDecoder(vgf_data);
  unique_ptr<vgflib::ModelSequenceTableDecoder> sequence_decoder =
      vgflib::CreateModelSequenceTableDecoder(
          vgf_data + header_decoder->GetModelSequenceTableOffset());
  unique_ptr<vgflib::ModuleTableDecoder> module_decoder =
      vgflib::CreateModuleTableDecoder(
          vgf_data + header_decoder->GetModuleTableOffset());
  unique_ptr<vgflib::ModelResourceTableDecoder> resource_decoder =
      vgflib::CreateModelResourceTableDecoder(
          vgf_data + header_decoder->GetModelResourceTableOffset());
  unique_ptr<vgflib::ConstantDecoder> constant_decoder =
      vgflib::CreateConstantDecoder(
          vgf_data + header_decoder->GetConstantsOffset());
  // Check the VGF decoders
  if (not(header_decoder && module_decoder && sequence_decoder &&
          resource_decoder && constant_decoder && header_decoder->IsValid() &&
          header_decoder->CheckVersion())) {
    ET_LOG(Error, "Failed to process VGF file internalsr");
    return false;
  }

  // Parse the sequences in the VGF (while there can be multiple sequences of
  // COMPUTE and GRAPH segments in the sequence, we currently expect a single
  // GRAPH segment to be present.
  const int segment_id = 0;

  debug_print_sequence(sequence_decoder);
  if (sequence_decoder->modelSequenceTableSize() != 1) {
    ET_LOG(Error, "Expected sequence length 1");
    return false;
  }
  if (sequence_decoder->getSegmentType(segment_id) !=
      vgflib::ModuleType::GRAPH) {
    ET_LOG(Error, "Expected segment to be of type GRAPH");
    return false;
  }

  // Extract first segment and it's associated module
  debug_print_modules(module_decoder);
  auto segment_name = string(sequence_decoder->getSegmentName(segment_id));
  auto segment_module = sequence_decoder->getSegmentModuleIndex(segment_id);

  auto segment_m_name = string(module_decoder->getModuleName(segment_module));
  auto segment_m_entrypoint =
      string(module_decoder->getModuleEntryPoint(segment_module));
  auto segment_m_spirv = module_decoder->getModuleCode(segment_module);

  // Build a shader from the module
  VkShaderModuleCreateInfo smci{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .codeSize = segment_m_spirv.size() * sizeof(uint32_t),
      .pCode = segment_m_spirv.begin(),
  };
  result = vkCreateShaderModule(vk_device, &smci, nullptr, &vk_shader);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to load shader from segment %d", segment_module);
    return false;
  }

  // Record our shader and entrypoint string
  vector<tuple<VkShaderModule, string>> shader_modules;
  shader_modules.push_back({vk_shader, segment_m_entrypoint});

  // Load our resource (tensors, constants) into their appropriate Vk objects
  vector<VkTensorDescriptionARM> descriptors;
  vector<tuple<VkTensorARM, VkTensorViewARM>> resources;
  vector<VkDataGraphPipelineConstantARM> constants;

  int IO_count = resource_decoder->size();
  for (int i = 0; i < IO_count; i++) {
    auto resource_type = resource_decoder->getDescriptorType(i).value_or(0);
    auto resource_format = vgflib::ToVkFormat(resource_decoder->getVkFormat(i));

    // Get tensor shape and strides
    auto shape = resource_decoder->getTensorShape(i);
    auto stride = resource_decoder->getTensorStride(i);

    switch (resource_decoder->getCategory(i)) {
      case vgflib::ResourceCategory::INPUT:
      case vgflib::ResourceCategory::OUTPUT: {
        // Expect IO to be a tensor type
        if (resource_type != VK_DESCRIPTOR_TYPE_TENSOR_ARM) {
          ET_LOG(
              Error,
              "Expected tensor type descriptor %u got %u",
              VK_DESCRIPTOR_TYPE_TENSOR_ARM,
              resource_type);
          return false;
        }

        // Allocate a tensor with backing memory
        VkTensorARM tensor;
        VkTensorViewARM tensor_view;
        VkDeviceMemory tensor_memory;
        VkTensorDescriptionARM tensor_description;
        result = allocate_tensor(
            vk_physical,
            vk_device,
            vgflib::ToVkFormat(resource_decoder->getVkFormat(i)),
            static_cast<uint32_t>(shape.size()),
            shape.begin(),
            static_cast<uint32_t>(stride.size()),
            stride.begin(),
            &tensor_description,
            &tensor_view,
            &tensor,
            &tensor_memory);
        if (result != VK_SUCCESS) {
          ET_LOG(Error, "Failed to allocate tensor for VGF resource %d", i);
          return false;
        }
        size_t e_size = get_format_size(
            vgflib::ToVkFormat(resource_decoder->getVkFormat(i)));
        if (0 == e_size) {
          ET_LOG(Error, "failed to get element size of VkFormat");
          return false;
        }

        bool is_in =
            resource_decoder->getCategory(i) == vgflib::ResourceCategory::INPUT;
        IOs.push_back(
            IO{vector<int64_t>(shape.begin(), shape.end()),
               vector<int64_t>(stride.begin(), stride.end()),
               e_size,
               tensor,
               tensor_view,
               tensor_memory,
               is_in});
        resources.push_back({tensor, tensor_view});
        descriptors.push_back(tensor_description);
        break;
      }
      case vgflib::ResourceCategory::CONSTANT:
        // Constants just need a descriptor
        descriptors.push_back(VkTensorDescriptionARM{
            .sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
            .pNext = nullptr,
            .tiling = VK_TENSOR_TILING_LINEAR_ARM,
            .format = vgflib::ToVkFormat(resource_decoder->getVkFormat(i)),
            .dimensionCount = static_cast<uint32_t>(shape.size()),
            .pDimensions = shape.begin(),
            // Note: stride_data of 0's causes size==0, null means stride==size
            .pStrides = (0 == stride.size() ? nullptr : stride.begin()),
            .usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM,
        });
        break;
      case vgflib::ResourceCategory::INTERMEDIATE:
        ET_LOG(Error, "Unsupported resource category INTERMEDIATE");
        return false;
      default:
        ET_LOG(Info, "Unsupported resource category UNKNOWN");
        return false;
    }
  }

  // Constants table - mapping of shader bindings to MRT's and their descriptors
  auto constant_indexes =
      sequence_decoder->getSegmentConstantIndexes(segment_id);
  for (uint32_t i : constant_indexes) {
    auto mrt_i = constant_decoder->getConstantMrtIndex(i);
    auto constant_data = constant_decoder->getConstant(i);
    constants.push_back(VkDataGraphPipelineConstantARM{
        .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CONSTANT_ARM,
        .pNext = &descriptors[mrt_i],
        .id = i,
        .pConstantData = constant_data.begin(),
    });
  }

  // Prepare our layout bindings from the segment's information
  vector<VkDescriptorSetLayoutBinding> layout_bindings;
  vector<VkDataGraphPipelineResourceInfoARM> data_graph_resources;

  auto set_count =
      sequence_decoder->getSegmentDescriptorSetInfosSize(segment_id);
  for (uint32_t d_idx = 0; d_idx < set_count; d_idx++) {
    auto handle =
        sequence_decoder->getDescriptorBindingSlotsHandle(segment_id, d_idx);
    auto binding_count = sequence_decoder->getBindingsSize(handle);
    for (int binding = 0; binding < binding_count; binding++) {
      auto binding_index =
          sequence_decoder->getBindingSlotBinding(handle, binding);
      auto MRT_index =
          sequence_decoder->getBindingSlotMrtIndex(handle, binding);
      auto MRT_type = resource_decoder->getDescriptorType(MRT_index).value();

      const VkDescriptorSetLayoutBinding layout_binding{
          .binding = binding_index,
          .descriptorType = vgflib::ToVkDescriptorType(MRT_type),
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_ALL,
          .pImmutableSamplers = nullptr,
      };
      layout_bindings.push_back(layout_binding);

      const VkDataGraphPipelineResourceInfoARM resource{
          .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM,
          // Note: we populate the resource_descriptors 1:1 with the MRT table,
          // so can directly use that index into the resource_descriptors
          .pNext = &descriptors[MRT_index],
          .descriptorSet = d_idx,
          .binding = binding_index,
          .arrayElement = 0,
      };
      data_graph_resources.push_back(resource);
    }
  }

  // create fixed layout for this module
  const VkDescriptorSetLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .bindingCount = static_cast<uint32_t>(layout_bindings.size()),
      layout_bindings.data(),
  };
  result =
      vkCreateDescriptorSetLayout(vk_device, &layout_info, nullptr, &vk_layout);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to create descriptor layout");
    return false;
  }

  std::vector<VkDescriptorPoolSize> poolSizes;
  poolSizes.reserve(layout_bindings.size());
  for (const auto& b : layout_bindings) {
    bool found = false;
    for (size_t idx = 0; idx < poolSizes.size(); ++idx) {
      if (poolSizes[idx].type == b.descriptorType) {
        poolSizes[idx].descriptorCount += b.descriptorCount;
        found = true;
        break;
      }
    }
    if (!found) {
      poolSizes.push_back({b.descriptorType, b.descriptorCount});
    }
  }

  // Create descriptor pool and descriptors for pipeline
  const VkDescriptorPoolCreateInfo descriptor_pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .maxSets = static_cast<uint32_t>(set_count),
      .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
      .pPoolSizes = poolSizes.data(),
  };
  result = vkCreateDescriptorPool(
      vk_device, &descriptor_pool_info, nullptr, &vk_descriptor_pool);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to create descriptor pool");
    return false;
  }

  const VkDescriptorSetAllocateInfo descriptor_set_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .pNext = nullptr,
      .descriptorPool = vk_descriptor_pool,
      .descriptorSetCount = static_cast<uint32_t>(set_count),
      .pSetLayouts = &vk_layout,
  };

  // Alloc descriptor sets
  // currently, as we require modelSequenceTableSize to == 1
  // we can only get one descriptor set.
  descriptor_sets.resize(layout_bindings.size());
  result = vkAllocateDescriptorSets(
      vk_device, &descriptor_set_info, descriptor_sets.data());
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to allocate descriptor sets");
    return false;
  }

  // write descriptor updates for every input
  auto input_slots =
      sequence_decoder->getSegmentInputBindingSlotsHandle(segment_id);
  auto input_size = sequence_decoder->getBindingsSize(input_slots);
  for (uint32_t i = 0; i < input_size; i++) {
    auto binding = sequence_decoder->getBindingSlotBinding(input_slots, i);
    auto mrt_i = sequence_decoder->getBindingSlotMrtIndex(input_slots, i);

    VkWriteDescriptorSetTensorARM write_desc = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM,
        .pNext = nullptr,
        .tensorViewCount = 1,
        .pTensorViews = &get<1>(resources[i]),
    };
    VkWriteDescriptorSet desc_set = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = &write_desc,
        .dstSet = descriptor_sets[0],
        .dstBinding = binding,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM,
        .pImageInfo = nullptr,
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };
    vkUpdateDescriptorSets(vk_device, 1, &desc_set, 0, nullptr);
  }

  // write descriptor updates for every output
  auto output_slots =
      sequence_decoder->getSegmentOutputBindingSlotsHandle(segment_id);
  auto output_size = sequence_decoder->getBindingsSize(output_slots);
  for (uint32_t i = 0; i < output_size; i++) {
    auto binding = sequence_decoder->getBindingSlotBinding(output_slots, i);
    auto mrt_i = sequence_decoder->getBindingSlotMrtIndex(output_slots, i);

    VkWriteDescriptorSetTensorARM write_desc = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM,
        .pNext = nullptr,
        .tensorViewCount = 1,
        .pTensorViews = &get<1>(resources[i + input_size]),
    };
    VkWriteDescriptorSet desc_set = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = &write_desc,
        .dstSet = descriptor_sets[0],
        .dstBinding = binding,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM,
        .pImageInfo = nullptr,
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };
    vkUpdateDescriptorSets(vk_device, 1, &desc_set, 0, nullptr);
  }

  // create our pipeline
  VkPipelineLayoutCreateInfo pipeline_layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .setLayoutCount = 1,
      .pSetLayouts = &vk_layout,
      .pushConstantRangeCount = 0,
      .pPushConstantRanges = nullptr,
  };
  result = vkCreatePipelineLayout(
      vk_device, &pipeline_layout_info, nullptr, &vk_pipeline_layout);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to create pipeline layout");
    return false;
  }

  // Shader Module Create
  VkDataGraphPipelineShaderModuleCreateInfoARM shader_info{
      .sType =
          VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM,
      .pNext = nullptr,
      .module = get<0>(shader_modules[0]),
      .pName = get<1>(shader_modules[0]).c_str(),
      .pSpecializationInfo = nullptr,
      .constantCount = static_cast<uint32_t>(constants.size()),
      .pConstants = constants.data(),
  };

  // Prepare Graph Pipeline
  VkDataGraphPipelineCreateInfoARM graph_pipeline_info{
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM,
      .pNext = &shader_info,
      .flags = VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT |
          VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT_KHR,
      .layout = vk_pipeline_layout,
      .resourceInfoCount = static_cast<uint32_t>(data_graph_resources.size()),
      .pResourceInfos = data_graph_resources.data(),
  };

  result = vkCreateDataGraphPipelinesARM(
      vk_device, // device
      VK_NULL_HANDLE, // deferredOperation
      VK_NULL_HANDLE, // VkPipelineCache
      1, // createInfoCount
      &graph_pipeline_info, // pCreateInfos
      nullptr, // pAllocator
      &vk_pipeline // pPipelines (VkPipeline*)
  );
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to create DataGraphPipeline");
    return result;
  }

  // prepare the graph pipeline session
  VkDataGraphPipelineSessionCreateInfoARM pipeline_session_info{
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM,
      .pNext = nullptr,
      .flags = 0,
      .dataGraphPipeline = vk_pipeline,
  };
  result = vkCreateDataGraphPipelineSessionARM(
      vk_device, &pipeline_session_info, nullptr, &vk_session);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to create DataGraphPipelineSession");
    return result;
  }

  // Allocate command buffer
  VkCommandBufferAllocateInfo buffer_allocate_info{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .pNext = nullptr,
      .commandPool = vk_command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};
  result = vkAllocateCommandBuffers(
      vk_device, &buffer_allocate_info, &vk_execute_cmd);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to allocate command buffers");
    return result;
  }

  // Allocate intermediates memory based on the pipeline requirements provided
  // by the driver
  VkDataGraphPipelineSessionBindPointRequirementsInfoARM
      bind_point_requirements_info = {
          .sType =
              VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENTS_INFO_ARM,
          .pNext = nullptr,
          .session = vk_session,
      };

  uint32_t bind_point_count = 0;
  result = vkGetDataGraphPipelineSessionBindPointRequirementsARM(
      vk_device, &bind_point_requirements_info, &bind_point_count, nullptr);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to get session bind point count");
    return result;
  }

  vector<VkDataGraphPipelineSessionBindPointRequirementARM>
      bind_point_requirements;
  bind_point_requirements.resize(bind_point_count);
  result = vkGetDataGraphPipelineSessionBindPointRequirementsARM(
      vk_device,
      &bind_point_requirements_info,
      &bind_point_count,
      bind_point_requirements.data());
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to get session bind point requirements");
    return result;
  }

  // Given the bind points, just make individual allocations and bind them
  for (const auto& bind_point_requirement : bind_point_requirements) {
    // These are the only allowed type and bindpoint with the current spec
    if (bind_point_requirement.bindPointType !=
        VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TYPE_MEMORY_ARM) {
      ET_LOG(
          Error,
          "Expected VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TYPE_MEMORY_ARM");
      return VK_ERROR_UNKNOWN;
    }
    if (bind_point_requirement.bindPoint !=
        VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM) {
      ET_LOG(
          Error,
          "Expected VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM");
      return VK_ERROR_UNKNOWN;
    }
    if (bind_point_requirement.numObjects != 1) {
      ET_LOG(Error, "Expected only one object for the bindpoint");
      return VK_ERROR_UNKNOWN;
    }

    VkDataGraphPipelineSessionMemoryRequirementsInfoARM memory_requirements_info = {
        .sType =
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_MEMORY_REQUIREMENTS_INFO_ARM,
        .pNext = nullptr,
        .session = vk_session,
        .bindPoint = bind_point_requirement.bindPoint,
        .objectIndex = 0, // NOTE: tied to numObjects assert above
    };
    VkMemoryRequirements2 memory_requirements;
    vkGetDataGraphPipelineSessionMemoryRequirementsARM(
        vk_device, &memory_requirements_info, &memory_requirements);

    VkMemoryPropertyFlags aims = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t memory_index =
        get_memory_index(vk_physical, memory_requirements, aims);

    VkMemoryAllocateInfo memory_allocate_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = nullptr,
        .allocationSize = memory_requirements.memoryRequirements.size,
        .memoryTypeIndex = memory_index,
    };

    VkDeviceMemory memory;
    result =
        vkAllocateMemory(vk_device, &memory_allocate_info, nullptr, &memory);
    if (result != VK_SUCCESS) {
      ET_LOG(Error, "Failed to allocate memory for intermediates");
      return result;
    }
    // so we can free this object in destructor
    intermediates.push_back(memory);

    VkBindDataGraphPipelineSessionMemoryInfoARM bind_info = {
        .sType =
            VK_STRUCTURE_TYPE_BIND_DATA_GRAPH_PIPELINE_SESSION_MEMORY_INFO_ARM,
        .pNext = nullptr,
        .session = vk_session,
        .bindPoint = bind_point_requirement.bindPoint,
        .objectIndex = 0, // NOTE: tied to numObjects assert above
        .memory = memory,
        .memoryOffset = 0,
    };
    result = vkBindDataGraphPipelineSessionMemoryARM(vk_device, 1, &bind_info);
    if (result != VK_SUCCESS) {
      ET_LOG(Error, "Failed to bind intermediates memory");
      return result;
    }
  }

  // Populate command once with our dispatch information
  VkCommandBufferBeginInfo beginInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(vk_execute_cmd, &beginInfo);

  // Sync what will be the data coming in from host
  VkMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
      .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
  };
  VkDependencyInfo dependency_info = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &barrier,
  };
  vkCmdPipelineBarrier2(vk_execute_cmd, &dependency_info);

  // bind pipeline + descriptor set
  vkCmdBindPipeline(
      vk_execute_cmd, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM, vk_pipeline);

  vkCmdBindDescriptorSets(
      vk_execute_cmd,
      VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM,
      vk_pipeline_layout,
      0, // first set
      1,
      descriptor_sets.data(), // descriptor set count + pointer
      0,
      nullptr // no dynamic offsets
  );

  // Dispatch the graph command
  vkCmdDispatchDataGraphARM(vk_execute_cmd, vk_session, nullptr);

  // Sync data back
  VkMemoryBarrier2 barrier_2 = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
      .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
  };
  VkDependencyInfo dependency_info_2 = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &barrier_2,
  };
  vkCmdPipelineBarrier2(vk_execute_cmd, &dependency_info_2);

  // end the command buffer
  vkEndCommandBuffer(vk_execute_cmd);

  return true;
}

bool VgfRepr::execute_vgf() {
  ET_LOG(Info, "Executing vgf");

  // Submit & wait for idle
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &vk_execute_cmd;
  VkResult result = vkQueueSubmit(vk_queue, 1, &submit, VK_NULL_HANDLE);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "VGF/VkCommandBuffer command submission failed");
    return false;
  }
  vkQueueWaitIdle(vk_queue);

  return true;
}

void VgfRepr::free_vgf() {
  vkFreeCommandBuffers(vk_device, vk_command_pool, 1, &vk_execute_cmd);
  vkDestroyDataGraphPipelineSessionARM(vk_device, vk_session, nullptr);
  vkDestroyPipeline(vk_device, vk_pipeline, nullptr);
  vkDestroyPipelineLayout(vk_device, vk_pipeline_layout, nullptr);
  vkDestroyDescriptorPool(vk_device, vk_descriptor_pool, nullptr);
  vkDestroyDescriptorSetLayout(vk_device, vk_layout, nullptr);
  vkDestroyShaderModule(vk_device, vk_shader, nullptr);
  for (int i = 0; i < IOs.size(); i++) {
    free_tensor(
        vk_device, IOs[i].tensor_view, IOs[i].tensor, IOs[i].tensor_memory);
  }
  for (auto memory : intermediates) {
    vkFreeMemory(vk_device, memory, nullptr);
  }
}

static uint32_t get_format_size(VkFormat format) {
  // Note: While this is a small subset of VkFormat, this supports all base
  //       types for tensors coming from the compiler flow. Tensor formats only
  //       specify single element type.
  switch (format) {
    case VK_FORMAT_R8_BOOL_ARM:
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_R8_SINT:
      return 1;
    case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16_SFLOAT:
      return 2;
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R32_SFLOAT:
      return 4;
    case VK_FORMAT_R64_SINT:
      return 8;
    default:
      ET_LOG(Error, "Unknown tensor format");
      return 0;
  }
}

} // namespace vgf
} // namespace backends
} // namespace executorch
