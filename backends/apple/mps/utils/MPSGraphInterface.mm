//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include "MPSGraphInterface.h"
#include "MPSGraphPackageExport.h"

namespace mps {

using namespace torch;
MPSGraphModule::MPSGraphModule() {
  TORCH_CHECK(macos_version_or_newer(MacOSVersion::MACOS_VER_14_0_PLUS),
  "MPS Executorch backend is supported only from macOS 14.0 and above.");

  mpsGraph = [MPSGraph new];
}

MPSGraphModule::~MPSGraphModule() {
  [mpsGraph release];
}

PyMPSGraphTensor*
MPSGraphModule::mpsGraphUnrankedPlaceHolder(MPSDataType dataType) {
  inputTensors_.push_back([mpsGraph placeholderWithShape:nil
                                                dataType:dataType
                                                    name:nil]);
  return inputTensors_.back();
}

PyMPSGraphTensor*
MPSGraphModule::mpsGraphRankedPlaceHolder(MPSDataType dataType, const at::IntArrayRef& shape) {
  inputTensors_.push_back([mpsGraph placeholderWithShape:getMPSShape(shape)
                                                dataType:dataType
                                                    name:nil]);
  return inputTensors_.back();
}

PyMPSGraphTensor*
MPSGraphModule::mpsGraphScalarPlaceHolder(MPSDataType dataType) {
  inputTensors_.push_back([mpsGraph placeholderWithShape:@[@1]
                                                dataType:dataType
                                                    name:nil]);
  return inputTensors_.back();
}

void
MPSGraphModule::set_outputs(py::args args) {
  for (const auto i: c10::irange(args.size())) {
    MPSGraphTensor* outputTensor = static_cast<MPSGraphTensor*>(pybind11::cast<void*>(args[i]));
    outputTensors_.push_back(outputTensor);
  }
}

PyMPSGraphTensor*
MPSGraphModule::mm(MPSGraphTensor* primaryTensor, MPSGraphTensor* secondaryTensor) {
  return [mpsGraph matrixMultiplicationWithPrimaryTensor:primaryTensor
                                         secondaryTensor:secondaryTensor
                                                    name:nil];
}

PyMPSGraphTensor*
MPSGraphModule::identity(MPSGraphTensor* inputTensor) {
  return [mpsGraph identityWithTensor:inputTensor
                                 name:nil];
}

bool MPSGraphModule::macos_version_or_newer(MacOSVersion version) {
  id mpsCD = NSClassFromString(@"MPSGraph");
  static auto compileOptions = [[[MTLCompileOptions alloc] init] autorelease];

  static bool _macos_14_0_plus = [mpsCD instancesRespondToSelector:@selector(imToColWithSourceTensor:descriptor:name:)] == YES;

  switch (version) {
    case MacOSVersion::MACOS_VER_14_0_PLUS:  return _macos_14_0_plus;
    default: return false;
  }
}

void MPSGraphModule::printGraph() {
  NSLog(@"%@", [mpsGraph debugDescription]);
}

MPSGraphExecutable*
MPSGraphModule::compileMPSGraphExecutable() {
  NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType *> *feeds = [NSMutableDictionary dictionary];
  for (const auto i: c10::irange(inputTensors_.size())) {
    feeds[inputTensors_[i]] = [[MPSGraphShapedType alloc] initWithShape:[inputTensors_[i] shape] dataType:[inputTensors_[i] dataType]];
  }

  NSMutableArray<MPSGraphTensor*> *targetTensors = [NSMutableArray new];
  std::for_each(outputTensors_.begin(), outputTensors_.end(), ^(MPSGraphTensor* outputTensor) {
    [targetTensors addObject:outputTensor];
  });

  MPSGraphExecutable *exec = [mpsGraph compileWithDevice:nil
                                                   feeds:feeds
                                           targetTensors:targetTensors
                                        targetOperations:nil
                                   compilationDescriptor:nil];

  return exec;
}

std::vector<uint8_t> MPSGraphModule::serialize() {
  MPSGraphExecutable* exec = compileMPSGraphExecutable();

  std::string dataFolder = "/tmp/";

  std::string name = "mpsgraphmodule_" + std::to_string(arc4random_uniform(INT_MAX));
  std::string mpsgraphpackagePath = dataFolder + name + ".mpsgraphpackage";
  NSString *mpsgraphpackageFileStr = [NSString stringWithUTF8String:mpsgraphpackagePath.c_str()];
  NSURL *bundleURL = [NSURL fileURLWithPath:mpsgraphpackageFileStr];

  MPSGraphExecutableSerializationDescriptor *serializationDescriptor = [MPSGraphExecutableSerializationDescriptor new];
  serializationDescriptor.deploymentPlatform = MPSGraphDeploymentPlatformMacOS;
  serializationDescriptor.minimumDeploymentTarget = @"14.0.0";
  [exec serializeToMPSGraphPackageAtURL:bundleURL descriptor:serializationDescriptor];

  NSString* mpsgraphpackage_manifest_file = [NSString stringWithUTF8String:(mpsgraphpackagePath + "/manifest.plist").c_str()];
  NSString* mpsgraphpackage_model_0_file = [NSString stringWithUTF8String:(mpsgraphpackagePath + "/model_0.mpsgraph").c_str()];

  NSURL* manifestPlistURL = [NSURL fileURLWithPath:mpsgraphpackage_manifest_file];
  NSURL* model0URL = [NSURL fileURLWithPath:mpsgraphpackage_model_0_file];

  NSData* manifest_plist_data = [NSData dataWithContentsOfURL:manifestPlistURL];
  NSData* model_0_data = [NSData dataWithContentsOfURL:model0URL];

  int64_t total_package_size = sizeof(ExirMPSGraphPackage) + [manifest_plist_data length] + [model_0_data length];
  ExirMPSGraphPackage *exirMPSGraphPackage = (ExirMPSGraphPackage*)malloc(total_package_size);
  assert(exirMPSGraphPackage != nil);

  exirMPSGraphPackage->manifest_plist_offset = 0;
  exirMPSGraphPackage->model_0_offset = [manifest_plist_data length];
  exirMPSGraphPackage->total_bytes = total_package_size;

  memcpy(exirMPSGraphPackage->data, [manifest_plist_data bytes], [manifest_plist_data length]);
  memcpy(exirMPSGraphPackage->data + exirMPSGraphPackage->model_0_offset, [model_0_data bytes], [model_0_data length]);

  std::vector<uint8_t> data((uint8_t*)exirMPSGraphPackage, (uint8_t*)exirMPSGraphPackage + total_package_size);
  free(exirMPSGraphPackage);

  return data;
}

} // namespace mps
