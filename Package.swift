// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260118"
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
    "sha256": "53e59a1bd235f91176caf24225b62723ebc1b392095d961f8b1fcb7a64e37227",
    "sha256" + debug_suffix: "5eb581cfa11b51829bc2a01a9aa26f07d7f824bc66ca9977d609dd541c51d56c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "cb27ea7a6aa9421ae4989aa77e74db780a06824ef3cbb4bfa481be9847e66cd8",
    "sha256" + debug_suffix: "af9665367589621fa1267342cf2e431cd3eb886c69c93fa814245d47e721adc7",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5e6bd548754366c0e77a71fea399e96d609b6e70803eebd0b1b76a9269b03f0a",
    "sha256" + debug_suffix: "1a42cc7af1c5b9d7584986accba86b2736efed5bb4d786bc08c90816beefb194",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "7c4384e35aec14da9c4e9954231a86d8e5d57f27d6d6950f247cb1642d3ea091",
    "sha256" + debug_suffix: "058a517b15d9423c81deb646c5b8228936118cd6e013131f113adc912a02c561",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "21ba77748b4f3ca0f01526e3ef0c9fe20fc2fe0a7edfb1382daec31a67630a9f",
    "sha256" + debug_suffix: "a5827162b965912608a584a78a7a438f3d63a74ca17565c08935c9892ec7372d",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "6ec5be58d9e02084ea7a93f20505f636e2adb9d3ab5b6aba32805e7da995c22b",
    "sha256" + debug_suffix: "503bd5e0ed969b2b58eb8225e0b1131d3ba9628f150acae5eb9c95ef339a43a0",
  ],
  "kernels_optimized": [
    "sha256": "5af2571fb5faa59939d79a74257e6a7ccb6ad3ac27c8d2e7b2638ea39372067d",
    "sha256" + debug_suffix: "483b992fa21a2ee4d23c56e4ab695eb55b24b07071c0b522440a76a296aeb401",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b47eec2f27cc1d9c5e0f515ce81b939824e9f94077cc817a3a942760319438b5",
    "sha256" + debug_suffix: "a6d504aa04d0b2d6a0a88dde2f023024f580792e36d0dabe634b3f4e29a3758f",
  ],
  "kernels_torchao": [
    "sha256": "b5927ba834b8224b23647792d23afbacf1588bde297145cf69ec7136a0f11144",
    "sha256" + debug_suffix: "5bc0eb903bca4ddf0f8857a22dbc822faa036970093fb5f2129bd2c5c67c6e0c",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "f13797bbe135f681a33ebf6b6b552bacd15db83e7b59f6c3d6995b755d0310ac",
    "sha256" + debug_suffix: "3ce29c57be8639904856ad2f5d96cea7d6e75787c81cb467ad673b81749cc015",
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
