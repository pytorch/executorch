// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250621"
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
    "sha256": "396465fb0506b748ab72ec19c22f2ab1bc5ae1c432e3b2840c1cd60d618e3db3",
    "sha256" + debug_suffix: "55fa02d1fa97cb6c86eaf976cbb5c5fc91e0ef7690aed583d2f4d839761164db",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "77c45b955f641729ff6add2e83f43d3bbafb9af53485bd1290723503d69b78d8",
    "sha256" + debug_suffix: "3b1827f560c619558c5da37b8ada6d464f3fc9bb5c4a938c1c99a256755ec6bc",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "68cd47b275c25135704ede401edfd32543d80a429f7a8d4233c7986118c4e37b",
    "sha256" + debug_suffix: "d118a8a58c457323d73f96d063c2238a43b79de0d55c985527003e1f822e060d",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "15baf59df61b2bbcd0f81e446e48529d321bf060095e3058cd71a593e014b8af",
    "sha256" + debug_suffix: "d6dc2e6c1402ed55f72da8fd631e74ee29baca1364950e75d7e671facb2bef30",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "08442084a26e6527b07848865aa017e109ea70a332f6dd0cf28819aafed0aa92",
    "sha256" + debug_suffix: "5455d1c0cef26aec723966638c8be0437b2db6f2fbb5b6c390dd31a5f3e6f057",
  ],
  "kernels_optimized": [
    "sha256": "b86e7e40e5ed2c5177cb3f10561c212eada20529acc5a8ce30407cb8b75b5a5a",
    "sha256" + debug_suffix: "890db1f35dcbc725ee38dede82b6921c2ee3dab8598913944c5a1274698823d0",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "cd45c6233c5bf82edc3475bb9ed88905875f30ffbb9e865f4068978eac346441",
    "sha256" + debug_suffix: "56fc3c377bf7f5c624ad10ba4881b0dccaf7b926449d4a85148c367fef2406e8",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "4b31620eca9209bf61cccb4ee5379c85ab159f0a806d5f88fd2c2f2635906085",
    "sha256" + debug_suffix: "1a71ffd658aab734dcdff29f86ad24d982881c26c66bfbcd1b109e437ef16900",
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
