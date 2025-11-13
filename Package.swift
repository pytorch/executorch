// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251113"
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
    "sha256": "b29b672eeb511fb99ad7f95190514301d061b06684682010099c5cb03d70adab",
    "sha256" + debug_suffix: "179a93bf7ca8146417312b5b77cc65b8a2a731ac37a4fbbfbe2cf07f215f24f7",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "f9388e3054df6201f00bc73b23917aedffc2202246f827a5385fec846a62beb1",
    "sha256" + debug_suffix: "bbcf864a2d572a5796d081c5958e24795a08a64b87805a6717181fcc819ff55d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d567fa97e1c9abeaa50c8b28949506de070e0d6cbf999939ec51d9eefca1f4bd",
    "sha256" + debug_suffix: "7f581c99f2b88772f6a3bf48d792055c6401645e2a9da41ee3c9b92b8bcdf5a0",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "4dddeaa8ce19867c8b84de7bdf55368e9a1a5e322128088b249bbdc9c52bc6e6",
    "sha256" + debug_suffix: "0418e924f6fe871a0bf37a73644b62af3f7f1a4511de98b3d21f7cfef28f0dd4",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "c705354ec7de9caf93d72b3eb7c0fd373f800de1702c542859ffdaa5db95d0ca",
    "sha256" + debug_suffix: "4b25d1c4c529bdab376195d962c6f9adad839ace7e582bd99b682781972fc83b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "525d61408bc1c3b417670b341e3773406e01e16d367bccc0f6ccffb6d04f0f5d",
    "sha256" + debug_suffix: "f0b72261882a2e70228e97defa8f46c28d8be69dfc797578aa92afbc49026cfe",
  ],
  "kernels_optimized": [
    "sha256": "0fdb6a52541e76a9815b19c40c6277966a72e3a4ff9d14353d4c5fed78f205f2",
    "sha256" + debug_suffix: "367ed807bbd25921a29a709c9f807421e84f3b5c24450621ff4eadb20a4a6085",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f0e6b95b42d7ea8ee15bf869982d2ad5e64e0d6dcea1624d40f14ed67a3a586a",
    "sha256" + debug_suffix: "877c39f0a63585a30409edc7e6bb68d9864789ec41b84e27965cfb04170583ac",
  ],
  "kernels_torchao": [
    "sha256": "d39cb814276052f1acbdd1e4d9ed1090739be183c2f57c5e5f8be6907375fe12",
    "sha256" + debug_suffix: "b9ff4fc2644b114bd7cb7595feccd7619e6c74221d9194f0fc8b1a04ed5619b3",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "ff3cdaef4b01d7d4d3190f685b2a925ff269dd9a861141fb0f378a07540c7ea1",
    "sha256" + debug_suffix: "876d5212a3acc109d163adb60a62a6b10a7ba47ce84eb5b8ddc883f9b3e41c7f",
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
