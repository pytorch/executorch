// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NOTE: This package manifest is for frameworks built locally with CMake.
// It defines dependencies and linker settings for Executorch components.
//
// To use prebuilt binaries instead, switch to one of the "swiftpm" branches,
// which fetch the precompiled `.xcframeworks`.
//
// For details on building frameworks locally or using prebuilt binaries,
// see the documentation:
// https://pytorch.org/executorch/main/using-executorch-ios

import PackageDescription
import Foundation

let debug_suffix = "_debug"
let dependencies_suffix = "_with_dependencies"

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

let packageProducts: [Product] = products.keys.map { key -> Product in
  .library(name: key, targets: ["\(key)\(dependencies_suffix)"])
}.sorted { $0.name < $1.name }

var packageTargets: [Target] = []

for (key, value) in targets {
  packageTargets.append(.binaryTarget(
    name: key,
    path: "cmake-out/\(key).xcframework"
  ))
}

for (key, value) in products {
  packageTargets.append(.binaryTarget(
    name: key,
    path: "cmake-out/\(key).xcframework"
  ))
  let target: Target = .target(
    name: "\(key)\(dependencies_suffix)",
    dependencies: ([key] + (value["targets"] as? [String] ?? []).map {
      key.hasSuffix(debug_suffix) ? $0 + debug_suffix : $0
    }).map { .target(name: $0) },
    path: ".Package.swift/\(key)",
    linkerSettings:
      (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
      (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
  )
  packageTargets.append(target)
}

// Test fixtures. add_coreml.pte and add_mul_coreml.pte are generated at CI
// time by extension/apple/ExecuTorch/__tests__/resources/generate_coreml_test_models.py
// (invoked by scripts/build_apple_frameworks.sh before `swift test`). They
// are gitignored, so include them in test resources only when present so
// that `swift test` runs on dev machines without CoreML python deps don't
// fail at the SwiftPM resolve stage.
let testResourcesDir = "extension/apple/ExecuTorch/__tests__/resources"
var testResources: [Resource] = [.copy("resources/add.pte")]
if FileManager.default.fileExists(atPath: "\(testResourcesDir)/add_coreml.pte") {
  testResources.append(.copy("resources/add_coreml.pte"))
}
if FileManager.default.fileExists(atPath: "\(testResourcesDir)/add_mul_coreml.pte") {
  testResources.append(.copy("resources/add_mul_coreml.pte"))
}

// SwiftPM resources must live under the target's path, so the ObjC test
// target uses copies of the canonical resources directory's fixtures. The
// copies themselves are gitignored and (re)created by scripts/build_apple_frameworks.sh.
let objcTestsDir = "extension/apple/ExecuTorch/__tests__/ObjC"
var objcTestResources: [Resource] = []
if FileManager.default.fileExists(atPath: "\(objcTestsDir)/add.pte") {
  objcTestResources.append(.copy("add.pte"))
}
if FileManager.default.fileExists(atPath: "\(objcTestsDir)/add_coreml.pte") {
  objcTestResources.append(.copy("add_coreml.pte"))
}
if FileManager.default.fileExists(atPath: "\(objcTestsDir)/add_mul_coreml.pte") {
  objcTestResources.append(.copy("add_mul_coreml.pte"))
}

let testLinkerSettings: [LinkerSetting] = [
  .unsafeFlags([
    "-Xlinker", "-force_load",
    "-Xlinker", "cmake-out/kernels_optimized.xcframework/macos-arm64/libkernels_optimized_macos.a",
    // CoreML backend registers itself with the global delegate registry via a
    // static initializer; -force_load ensures that initializer is pulled in so
    // the CoreML-delegated test fixtures can actually instantiate the backend.
    "-Xlinker", "-force_load",
    "-Xlinker", "cmake-out/backend_coreml.xcframework/macos-arm64/libbackend_coreml_macos.a",
  ])
]

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v12),
  ],
  products: packageProducts,
  targets: packageTargets + [
    .testTarget(
      name: "tests",
      dependencies: [
        .target(name: "executorch\(debug_suffix)"),
        .target(name: "kernels_optimized\(dependencies_suffix)"),
        .target(name: "backend_coreml\(dependencies_suffix)"),
      ],
      path: "extension/apple/ExecuTorch/__tests__",
      exclude: ["ObjC", "resources/generate_coreml_test_models.py", "resources/.gitignore"],
      resources: testResources,
      linkerSettings: testLinkerSettings
    ),
    .testTarget(
      name: "objc_tests",
      dependencies: [
        .target(name: "executorch\(debug_suffix)"),
        .target(name: "kernels_optimized\(dependencies_suffix)"),
        .target(name: "backend_coreml\(dependencies_suffix)"),
      ],
      path: "extension/apple/ExecuTorch/__tests__/ObjC",
      exclude: [".gitignore"],
      resources: objcTestResources,
      linkerSettings: testLinkerSettings
    )
  ]
)
