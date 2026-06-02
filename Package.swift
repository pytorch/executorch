// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260602"
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
    "sha256": "1da63c1c18c0489c4ebd1aeb842f9b55f89872dabf6f3cdb981938d3b4789eff",
    "sha256" + debug_suffix: "cc2579f51e0efda74f85c6ebcaf86bbea15bd67207d83fe027d334bf93260d01",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "46abc6b657c90e303644ba424f9285d26e5f1686ce51e5a88f60d315e3d09f3f",
    "sha256" + debug_suffix: "91ce91a9976ea75726b23f7de05bb2bc45f5688dac107a073edba2b4be21c84a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3c3afd6b6135f915190f94420691fa60764cd3dccc312b2f661b8023fd1b26d6",
    "sha256" + debug_suffix: "f9f03a3dedba22307cece204226778ee0e6ca3b481b837581f132ad0b151bd23",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "a67777ab5967b19394caf45ea26def28d6f5feaeed8aaadfcb52e1fbc4bd40f2",
    "sha256" + debug_suffix: "2636a790e82453538a0184cd0cb609e00e9ee37d7f596bd338cb4c5c8aee8c77",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "100f3344c333c1e27cb102d43b72fddbb57d3ac8bd5be6d6fbaa756f5224bcd5",
    "sha256" + debug_suffix: "f69793b0f23a1b7cef68bd870aec72a54b5be6b8192185c5deef3931c9af3b58",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "45772f1b2fbe1637b076d897c02085a6643833c65442e0656e5a2ae4b3695807",
    "sha256" + debug_suffix: "6de9565b701bb1863e60f145c24b6c7e705b9796522b1adcd7f6559649202a2b",
  ],
  "kernels_optimized": [
    "sha256": "2ef2ace65c298d3ef89740d9297fa8ba1618550d93bba7b14c8ae60391057089",
    "sha256" + debug_suffix: "33250ff08b6b9329c7aa69f2632c9babe3ca0590cd1a100d4c7b4e7a5b3ad5bd",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "cebda13e272147f4549c231f811f1f3e6a614c95db5f8efcffea594b4a3104a9",
    "sha256" + debug_suffix: "54ee142c3371de600e9fe399c2228ebe5e0d04b155bbe2d9318934121a6f23e1",
  ],
  "kernels_torchao": [
    "sha256": "11a97c9b9b5e6d21ec65291689f8f715f1eeff09d44a5f0568970e83c2bc9e52",
    "sha256" + debug_suffix: "0cce87f0ab3dadd3326157fe9d5c99e1dd6fdc836eda92d1af7c2709052457af",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "c73610e2a71b54763626049cc9312bb55683c1f1c4ac994509f570ae8a223260",
    "sha256" + debug_suffix: "5d578c4431284422378ae3499ea940b2910166682c96ac5cc7f7ade2ab6c783f",
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
