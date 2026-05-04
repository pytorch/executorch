// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260504"
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
    "sha256": "1f79440ccd3b70017d4c5896e667667e46230b6d4fc33a43b22e74caf208d94f",
    "sha256" + debug_suffix: "c3afbe5ce83810b536cb1befb9f8ac7ef0641eae2248e743d25d00501e3fb3ad",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "bcf219355f4acd010ed152bf7c582d4c79a58fe9c15b6bb26e9a2e3b09e7f317",
    "sha256" + debug_suffix: "aee6ac356eb21b4cce548ca49c4f08a7c6baec44f5112e64eb7d411261011c30",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a2f320684ca6ced0082af32791fa7eed79a5cc9cb34ab6baf8cbc6d692d44322",
    "sha256" + debug_suffix: "3b81aa3efce3ac5af0cc706219c4bacdc1ee72cbc594f12f732e9aded11e1d01",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d8214b7a20fa19fadf6fa5c258e25b43bb652e464fc1b94c4d253b6c78e72a64",
    "sha256" + debug_suffix: "f87a99d91d9b0ab26150cc91d1c2ceb0aa3ac1acfe4c52f2067e9e4e5c9027bb",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "221c67e17a9842eab04a35414469507d2e0cdfcce6f674023c84a106acc81b74",
    "sha256" + debug_suffix: "11bd513e9f4001d19a2a604f9099bdf1fdfccde1c8aa7607e8803a09318b104c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "64f0f6ea6659bf47594cd1f27cee7b7f0df54125a814f1ae02b8e7688dedac33",
    "sha256" + debug_suffix: "48eea2620e70c43c6c9fcb7af3a831c812cb3e0fcc581f5df6dcdf03381c60c5",
  ],
  "kernels_optimized": [
    "sha256": "19b3090144ab2c2e1c08e7cd0f7d84f577ec4062494b875abe962220cfcacbc8",
    "sha256" + debug_suffix: "cb27c403aa130955c52afda83f73f1fa25b69d3990673ce211fc86178acd524a",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d9e612f2040c5f5fff56f94b591c88c2a0722a95285c696319e12e9638157581",
    "sha256" + debug_suffix: "10801392b589bba5bdfd5db698a0dbdad582efb7ad30da0c57dd847fd2e693b1",
  ],
  "kernels_torchao": [
    "sha256": "ac5092fd574e884cc028ee69bd51c426b19f133ca3ebe1428324e4b128d82b80",
    "sha256" + debug_suffix: "7f7d0c4ddb8c145524082793f98e6ac14c16180720fa0b4af2cebc35ffa56016",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "3425aaf7ddb00f64ff2e88f0166607bdcf9d7b6f7d91397bf166bdd355676923",
    "sha256" + debug_suffix: "0e12810161b3d512715469dddac4c81f1c3332c1134d35a173789216844aaef4",
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
