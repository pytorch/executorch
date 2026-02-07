// swift-tools-version:6.2
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NOTE: This package manifest is for binary artifacts built locally with CMake.
// It defines dependencies and linker settings for executorch components.
// On Apple platforms it uses .xcframeworks; on Linux/Windows it expects .artifactbundle outputs.
//
// To use prebuilt binaries instead, switch to one of the "swiftpm" branches,
// which fetch the precompiled `.xcframeworks`.
//
// For details on building frameworks locally or using prebuilt binaries,
// see the documentation:
// https://pytorch.org/executorch/main/using-executorch-ios

import Foundation
import PackageDescription

let debug_suffix = "_debug"
let dependencies_suffix = "_with_dependencies"
let trait_cpu = "CPU"
let trait_cuda = "CUDA"
let trait_migraphx = "MIGRAPHX"

func deliverables(_ dict: [String: [String: Any]]) -> [String: [String: Any]] {
  dict
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      result[key] = value
      result[key + debug_suffix] = value
    }
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      var newValue = value
      if key.hasSuffix(debug_suffix) {
        for (k, v) in value where k.hasSuffix(debug_suffix) {
          let trimmed = String(k.dropLast(debug_suffix.count))
          newValue[trimmed] = v
        }
      }
      result[key] = newValue.filter { !$0.key.hasSuffix(debug_suffix) }
    }
}

let products = deliverables([
  "backend_coreml": [
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "libraries": [
      "c++",
    ],
  ],
  "onnxruntime": [
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [:],
  "kernels_optimized": [
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [:],
  "kernels_torchao": [
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [:],
])

#if os(Linux)
let linuxSupportedProducts = Set([
  "backend_xnnpack",
  "executorch",
  "executorch_llm",
  "kernels_llm",
  "kernels_quantized",
  "kernels_torchao",
  "onnxruntime",
].flatMap { [$0, $0 + debug_suffix] })

let activeProducts = products.filter { linuxSupportedProducts.contains($0.key) }
let activeTargets = targets
#else
let activeProducts = products
let activeTargets = targets
#endif

let packageProducts: [Product] = activeProducts.keys.map { key -> Product in
  .library(name: key, targets: ["\(key)\(dependencies_suffix)"])
}.sorted { $0.name < $1.name }

#if os(Linux)
let linuxArtifactsRoot = ProcessInfo.processInfo.environment["EXECUTORCH_SWIFTPM_LINUX_ARTIFACTS_DIR"] ?? "cmake-out"
let linuxVariant = ProcessInfo.processInfo.environment["EXECUTORCH_SWIFTPM_LINUX_VARIANT"] ?? ""
let linuxStdlib = ProcessInfo.processInfo.environment["EXECUTORCH_SWIFTPM_LINUX_STDLIB"] ?? "stdc++"

func binaryPath(_ name: String) -> String {
  let variantPart = linuxVariant.isEmpty ? "" : "/\(linuxVariant)"
  return "\(linuxArtifactsRoot)\(variantPart)/\(name).artifactbundle"
}

func binaryPath(_ name: String, variant: String) -> String {
  "\(linuxArtifactsRoot)/\(variant)/\(name).artifactbundle"
}

func normalizeLibraries(_ libs: [String]) -> [String] {
  libs.map { $0 == "c++" ? linuxStdlib : $0 }
}
#elseif os(Windows)
let windowsArtifactsRoot = ProcessInfo.processInfo.environment["EXECUTORCH_SWIFTPM_WINDOWS_ARTIFACTS_DIR"] ?? "cmake-out"

func binaryPath(_ name: String) -> String {
  "\(windowsArtifactsRoot)/\(name).artifactbundle"
}

func normalizeLibraries(_ libs: [String]) -> [String] {
  libs.filter { $0 != "c++" }
}
#else
func binaryPath(_ name: String) -> String {
  "cmake-out/\(name).xcframework"
}

func normalizeLibraries(_ libs: [String]) -> [String] {
  libs
}
#endif

var packageTargets: [Target] = []

for (key, _) in activeTargets {
  packageTargets.append(.binaryTarget(
    name: key,
    path: binaryPath(key)
  ))
}

for (key, value) in activeProducts {
  packageTargets.append(.binaryTarget(
    name: key,
    path: binaryPath(key)
  ))
  var dependencies: [Target.Dependency] = ([key] + (value["targets"] as? [String] ?? []).map {
    key.hasSuffix(debug_suffix) ? $0 + debug_suffix : $0
  }).map { .target(name: $0) }

#if os(Linux)
  if key == "onnxruntime" || key == "onnxruntime\(debug_suffix)" {
    let cudaTarget = key.hasSuffix(debug_suffix) ? "onnxruntime_cuda\(debug_suffix)" : "onnxruntime_cuda"
    let migraphxTarget = key.hasSuffix(debug_suffix) ? "onnxruntime_migraphx\(debug_suffix)" : "onnxruntime_migraphx"

    dependencies.removeAll { dependency in
      if case let .target(name, _) = dependency { return name == key }
      return false
    }
    dependencies.append(.target(name: key, condition: .when(traits: [trait_cpu])))
    dependencies.append(.target(name: cudaTarget, condition: .when(traits: [trait_cuda])))
    dependencies.append(.target(name: migraphxTarget, condition: .when(traits: [trait_migraphx])))
  }
#endif

  let target: Target = .target(
    name: "\(key)\(dependencies_suffix)",
    dependencies: dependencies,
    path: ".Package.swift/\(key)",
    linkerSettings:
      (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
      normalizeLibraries(value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
  )
  packageTargets.append(target)
}

#if os(Linux)
let linuxVariantTargets: [(name: String, variant: String)] = [
  ("onnxruntime_cuda", "cuda"),
  ("onnxruntime_cuda\(debug_suffix)", "cuda"),
  ("onnxruntime_migraphx", "migraphx"),
  ("onnxruntime_migraphx\(debug_suffix)", "migraphx"),
]

for target in linuxVariantTargets {
  packageTargets.append(.binaryTarget(
    name: target.name,
    path: binaryPath(target.name, variant: target.variant)
  ))
}
#endif

#if os(Linux)
let testTargets: [Target] = []
#else
let testTargets: [Target] = [
  .testTarget(
    name: "tests",
    dependencies: [
      .target(name: "executorch\(debug_suffix)"),
      .target(name: "kernels_optimized\(dependencies_suffix)"),
    ],
    path: "extension/apple/ExecuTorch/__tests__",
    resources: [
      .copy("resources/add.pte"),
    ],
    linkerSettings: [
      .unsafeFlags([
        "-Xlinker", "-force_load",
        "-Xlinker", "cmake-out/kernels_optimized.xcframework/macos-arm64/libkernels_optimized_macos.a",
      ])
    ]
  )
]
#endif

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v12),
  ],
  traits: [
    trait_cpu,
    trait_cuda,
    trait_migraphx,
    .default(enabledTraits: [trait_cpu]),
  ],
  products: packageProducts,
  targets: packageTargets + testTargets
)
