// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260819"
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
    "sha256": "b0a1f5a30776f3e2c094aee2e8321224a23a1aea1095014ff00d29a2adbd3653",
    "sha256" + debug_suffix: "24eace8e830ed444a62912ffd0b56bbce0483382718bbc31d28944aa531bb647",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5f3824755f800e530cb225b78e24b39c7db3d9e35e3b0e26989e72210698915a",
    "sha256" + debug_suffix: "a40307b2be61c9fa3e0c0407d093fd12cbb659317ca15e9d8195b1ac11197ea4",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f71ff6f5cb2b0bad3252e98a3bf6cc3cd209ad2e1c5fc400f13d92fe9363e1f3",
    "sha256" + debug_suffix: "fdf7e127755a45f178b7ff41f26cbf510eb185efefa079206ddcfcd7e0d37031",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "2630ffb74d503e9083e42432fb775d935274c2e3bf6627c22f3a656f2e811838",
    "sha256" + debug_suffix: "08ad6b999ea16a1093e193d9949942a9e909593f0885b4f571473e85e80dcd5a",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "69577dcafcc2241a590aab67fe593cd6b567b4a1306cef213435e7c08e178dd3",
    "sha256" + debug_suffix: "890af75040a570058815229088f305bfb46ee25e63ac216d2bcd36e3c1a6ff92",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "0448f7957d72f0f30474e5814f105213e68af79440f5b1ead22b60f4c2eea445",
    "sha256" + debug_suffix: "0ea99807354600ba0ea5b8912175525834e3e2d15d5914f40d91074f832f0ca4",
  ],
  "kernels_optimized": [
    "sha256": "e9f70a0333e9b686d8030d4dda375562eaf8f08e6d962e1b6f7ec6996f966d70",
    "sha256" + debug_suffix: "0c3290b331581e1abd223582ec907378f04983dafea5a7c75e553f25fc5c4ce4",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "0dbb6999206a46a4094a23e8ab253af6b9bd389208d0242c3c816dfb7f0f9a28",
    "sha256" + debug_suffix: "2386a0f44a3c892ac16fb7d7875cf137b4de6e9b8c23358beb55edc3d96fbfa8",
  ],
  "kernels_torchao": [
    "sha256": "c4ebb4e4bf420ee4cb2ad00e8f31da4705bd36ac86e454b07aec1ae01083e422",
    "sha256" + debug_suffix: "7b36c141c9fff4507ae7680699839a9ff4575f5899ade857e265a90ce83c6a2f",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "9d8f3f9ecc4ab9579ec32a4ccd5d5363ef385a82ba62eca7b0d060b4876cbe6f",
    "sha256" + debug_suffix: "70aa9389fab4f409ba2e8a9898e01034f121c01e74a3cb5f61a65e11e345d113",
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
