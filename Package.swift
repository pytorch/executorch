// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251107"
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
    "sha256": "8f576d7e64eb541769fdc1648a2855e64e4be7660b11f02cba60eeeb311ebca5",
    "sha256" + debug_suffix: "4154224b8b5821987a3edc1e1b6f635674532e2e151f30646b7455277c70faba",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "17f3cdecff0e6611504339def7aa9fd4261dc3476c60064c9a8027c31109dcd6",
    "sha256" + debug_suffix: "d1c82aa4b947691c7428d2745aceff13f1d29427763df6299b0eee1c1d68c821",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d89295b665b7f133126e4d1e320bfe6dd693c43a17485b83728c7fbbfb1c4a09",
    "sha256" + debug_suffix: "8fc81df4475b502aa0c083346f5ec50a0f12b3a9d1a4e8e7267bcc8561a72873",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5b3e8c60a3443e659c5229712e14f1872a0d32848f845ee17c0a277b5337446e",
    "sha256" + debug_suffix: "abf1ad9b9435f43ea3f5b5cae8bdbe2b3ca37296430464bca4f184022aceec17",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "cd9742c4b75db059275ba10462dc0c3932303ceb340e7df81431ba2a8a95056e",
    "sha256" + debug_suffix: "592b5b6e6dea32c591626c4752e05791585fc7ec02ec2b1d6c287fa5243ac086",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "09a8bd4e0bbcdea99a0fca34f1b72759db4c147f1f3890e0b2c9ccae8ce9b01c",
    "sha256" + debug_suffix: "75571099e605109d9315d1ba0c6d51ecad10191fd69f5ecdab6dbfea6b8d84e5",
  ],
  "kernels_optimized": [
    "sha256": "621d90a2447cfe1c2d91f079a912cddd2848a9536d71abfd7534412b5baefd76",
    "sha256" + debug_suffix: "6d8aaa25db9b4b26492a94bc9e51c4a06760db0747b568ccebebdcd0f6dec4bf",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9872138b6db86aaba37f32ba14771cfa8d13ff590fe8675de1130d9e28c32355",
    "sha256" + debug_suffix: "0f4c616abc53d7f41cd93d75b530d465be00984ad10508621f9ed43eb104d34d",
  ],
  "kernels_torchao": [
    "sha256": "f177a89a11f817bbc8d753a7428506c44b6eb57e946232ed5915b63d1de81490",
    "sha256" + debug_suffix: "8fb71c5a80bbc4318bde5347beec399adf2f1700f4ae9683af43ff358f89f9a7",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "549af2bd1e195aa55e96d8f506bc39656b1328f2a6a4cc6e744695652d0fff18",
    "sha256" + debug_suffix: "d7f731b305c8abbd6c9a7ec3635de31fa2c55a08d331f55e07e5a6b036334439",
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
