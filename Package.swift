// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250719"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
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
    "sha256": "bbaffbcee9684c6fd642e8b0fcb0312d28c8c80c2e79f19dd86f4fc12ebb59b1",
    "sha256" + debug_suffix: "f7f584fd49d483d0cfa43385aaa6ae7351153e1498ffec11583b5a74f990129f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ec7d8e26ceb0a0b04227138e92cb17c845ea07f4a83ebfe118a252a82c319896",
    "sha256" + debug_suffix: "c1151ff4b44eb2f78a650179a7e65bb0c954fc3560d1735fc1f3fea8463355be",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b21e4f5c283b93757df16d26aac6c5247d92e2f677d7a4a728310eacfbfbe649",
    "sha256" + debug_suffix: "4144b78f22f33c9e47e77fe543e01384dd7662ae166c8ca59fbfc8b0ba7fdc47",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "952a6d8d2e26cd93b25c8f910c8bda51568eb7a7101ab2e46d192a75176fb51e",
    "sha256" + debug_suffix: "d1b3b4654058e571b677a33ca0c267376f8bb624cdc576fa594229a686610a6a",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a0e8e80aa2ea3c8a3757de39ed05b8bb54873d176c45e2a7395441e5f7a7f999",
    "sha256" + debug_suffix: "ef2dfdfe7d09d6acb2aa309377c852ee60e51deff895f1a1ed699e3f84ad2457",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c775faa3fb58c52a57eb26aedd57e90f8fa21467247d54c3450add897fb86b7f",
    "sha256" + debug_suffix: "f4548b86b1088c9cbf0a2ab00faca88bb85d712f4f9cb544803ba9fee30711b4",
  ],
  "kernels_optimized": [
    "sha256": "f497ee194edd07eb254bc00d08cf046ba3c33adebbe048f9ae2fe87e5eadc5d5",
    "sha256" + debug_suffix: "f82152823411d6b5683184b4c3fce4af4d9406efc919b9ba7c8936bc063c83e9",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "5dcfb720c50d1df98c55ccbfedd5fb3a947fe2acffa88b242ae4eb3f5a59bff7",
    "sha256" + debug_suffix: "c46422565a219d2ab6c4ef227f4f9925f488386e0ebbe3cad4d4218f5cbc4f42",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "717ae7b181a0f7f11ad0cc2d2b61ab9b0f53869ce4fc0e2c9f547167060af43a",
    "sha256" + debug_suffix: "2fe8fa7e63af4c2ea74b2571178036d8dbe72972fd844045d0f38df02eea21a6",
  ],
])

let packageProducts: [Product] = products.keys.map { key -> Product in
  .library(name: key, targets: ["\(key)\(dependencies_suffix)"])
}.sorted { $0.name < $1.name }

var packageTargets: [Target] = []

for (key, value) in targets {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
}

for (key, value) in products {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
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

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v12),
  ],
  products: packageProducts,
  targets: packageTargets
)
