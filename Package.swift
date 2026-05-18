// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260518"
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
    "sha256": "73b2016c32727347a5d8f4b3fc17cc061760fcd3f36e377a36594009f334056d",
    "sha256" + debug_suffix: "f34dcaf2ff3458ccd31bfdf6501ca3a614cc03dd165492ba5d0a1d0aa25f781f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2640c2bd11f00c7019e0fc300f34d5e255df17a09e5a1ec804a1f2f018f1aebf",
    "sha256" + debug_suffix: "286a9f755b1920d1f28f0cf421c26d7ac04ff25ad0f3252c0357dad0236d9ef2",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "fa1306ce2cda0eb97f053cec9f3aa298ca3130ba9d0d78f340a63083f3855bf0",
    "sha256" + debug_suffix: "1cff89e892f19b25165188dc41046db012f911983188bbee3b1d9fa42f68b2c8",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "87f3a4fa27fe0d97194f4ab2c60622d0382cb44a11f984add54d5d698ecabb8e",
    "sha256" + debug_suffix: "735037b84814bd329631ac90457a3a4b4576a3a6c7e2ea2bd8892adb111d2579",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "b02e528163ab01a7e28f0de3b9e53075f059ab894535715ba730fe008dcffffc",
    "sha256" + debug_suffix: "a28bc757ac6b9e97ed659b94785a3ce4a44a105906dd0c659be4aa3363b86eeb",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "4e95cbe666a181fbcc252f080f30019138cca50d7b682cfdc275f5863cb7d1e9",
    "sha256" + debug_suffix: "8ccddee25ca35e1d77750271ccd92a00ea7458a976a98fc01ff2a0fdadec9fae",
  ],
  "kernels_optimized": [
    "sha256": "0e202b85c81a9e7cd6657ad8b51b431310cdb9d39d9903e0ec42242debe550db",
    "sha256" + debug_suffix: "b00d8e727dc485734485c01f449327bc3811f398673db90a027102627e3b8f03",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "7c8572efe11d30e2821f7611d59ffbf5cc2d6e9c9ec96236b275d14a80c587dc",
    "sha256" + debug_suffix: "609a4cbf08d9e9b523d2a5f497bc1e5df6e7dde2304e3cec8706d7f9c1af332b",
  ],
  "kernels_torchao": [
    "sha256": "bc6e71fd85071e26f2377b92ffdf076b7b509dd61c190a137baf3cadb04bd420",
    "sha256" + debug_suffix: "34e31bbf6cd8a9c5985f02742621e342a8559b0054168c53c69f91a69b23ebff",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "7c4e65ca173d7156e2aecddcd6b2fc3436201f10856fdb036001039b88eb56f0",
    "sha256" + debug_suffix: "4f995abc0dfbbcb40316a1163f5ce91e5764c1e86c4ffa590fcaf85153e217c1",
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
