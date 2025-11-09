// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251109"
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
    "sha256": "70be57421082b5956b13dbfa71b4351182e54c89de376f763e55b8f16235f682",
    "sha256" + debug_suffix: "a20388f34ecba342a23f1a5de605bcc787bd6015e3a7675eecc7495225105ef8",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2c5296f7e421ae4bde7043ef7acda2710c9f560ce7620b755614a933e96ad3d9",
    "sha256" + debug_suffix: "72b8d33498bd075a1ad82c795531ce1117c3bcd925c2bc9203365d10f86288aa",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "181a54791416b6bd7f1b5d91a752b637f4205ccb74218969e37dc3a2c09214f5",
    "sha256" + debug_suffix: "58a88dcb9401a8c97ce6362d324031252f63f5db302c0a5d05119adc386691c6",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "80b8ea8d36a5886b451ed81a3c0cfa8cb6f7914c439c9fbcb1acf88b105dcbe1",
    "sha256" + debug_suffix: "d03e5d3ba4f9e2ba7dcf4e4c8237214b763dcf7fc34e3f209040af0f4ccf573c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "99d55312c63106061b621d9fbc580e0978f555890120efe2b72e11544e74471d",
    "sha256" + debug_suffix: "8045641c34248b66c13aebfc7aa5f7a7c74a5bc1cb271b1935c7a8f4c277a7a5",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f31eeb49b7ada9aa92ec9d7d4e05d0cfae1fcc505d518aecaa29a444c1b6698e",
    "sha256" + debug_suffix: "7df8d4c1761241cdcc992ebbf0b8e5eb70c7f44dd4c382461bfb290b31721be1",
  ],
  "kernels_optimized": [
    "sha256": "e7d1902995aaaf26cd7706fd7a74b6c7b98217fdc2c7b526f1904bfe1ced8dce",
    "sha256" + debug_suffix: "0e3228e348c081136aa1e8571235969b0032cbd83e9fd9e3a071aaa15a3010d6",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "881f2d004acb2c4e5e08528e1f0b4e08ba68fea45785634cc2efc012c673df9b",
    "sha256" + debug_suffix: "8b53b01582e25bc6a106a5c30b1aadc466b00a32bc4f33bff8f2830dddfff37c",
  ],
  "kernels_torchao": [
    "sha256": "ad17500cf0f5ff6967505fea25f767c6471c81f1060eb408a41bd17e7a8232d6",
    "sha256" + debug_suffix: "a85ecad5315dc946fec0dc4dec98dab728f5b7be3c34713197fedc4d81a68865",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "17df5ba27319b9b24350ef84d212c65e0fa457fe3d34cb074982912e6d7b9042",
    "sha256" + debug_suffix: "3b98ebb9bdee2a65b0784415af48eece7a69d0ee9989ed07c5db14d9a0ed8f5e",
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
