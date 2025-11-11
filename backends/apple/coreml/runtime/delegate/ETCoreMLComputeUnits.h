//
// ETCoreMLComputeUnits.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

/// An enum representing the compute options on the system.
typedef NS_OPTIONS(NSUInteger, ETCoreMLComputeUnits) {
    ETCoreMLComputeUnitUnknown = 0, // Unknown compute unit.
    ETCoreMLComputeUnitCPU = 1 << 0, // Represents the CPU compute unit.
    ETCoreMLComputeUnitGPU = 1 << 1, // Represents the GPU compute unit.
    ETCoreMLComputeUnitNeuralEngine = 1 << 2 // Represents the NeuralEngine compute unit.
};
