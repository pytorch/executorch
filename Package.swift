// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250926"
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
    "sha256": "3076fdf8185ac04dcb4778e8447e1ecd42c0cd8c868083094286291f7d7f0125",
    "sha256" + debug_suffix: "d9b164497c7594e1d792faf9237e80fe2bcade661e2b56ea29ae20d560db1969",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "64dd7923349aa41db33af7c6b7eb514f99a7d4a70d246996018a034da19c6e60",
    "sha256" + debug_suffix: "ac45bdd7b568514cfc982223839a48e2d2962ab70d14628b3089ddbbdb8baeda",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2c8904506cc4e753bbe30e84c6a5685fc4e2befa504ce384e9df42c465ee67aa",
    "sha256" + debug_suffix: "260d579eae8a0ec96907beaf7dfe8049841dd047f3fc57ee0f2f0ecfb9dadb9c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "fe239ac578f08ec80a8bced2aebbe91cda38c51ece3fbeaf2620021ffb16b268",
    "sha256" + debug_suffix: "fa32a465e4ee222aabefb48827b600a7d11a6a3ddaa35e3c12b85e8dfd382a28",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "c055e0695498cdddd8e03c2e5a10f5afa0c2c5204895fe94251438b214240778",
    "sha256" + debug_suffix: "7c02432274ca9ab09dbc6b963256663765b60d748cb86c8e9552c7651b2e4945",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "3f1ede2ecd3057d1e4fd71d546d8b627548317e2553d18bff3b58ce91f3a67d1",
    "sha256" + debug_suffix: "c6d63768fd8c765f2922863f5f84f78d78ee2ed975071eac9ceedfbcfb56331f",
  ],
  "kernels_optimized": [
    "sha256": "5f1778034e9ab8b51cbe0722703b7c16f586e9ce12d83f94084d2463d776bd90",
    "sha256" + debug_suffix: "14146893bfb6ebe1871e14c7ef86155ce87c02652e3be1de5f48ad0a89a3d14d",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "5fd58b5e87501060751f5912fa301debb16992b4d957d2c81d31ae1061b1b284",
    "sha256" + debug_suffix: "698e5f1ba0892dcd474b76bfe50d317b2243581bab3311525b324248daedadad",
  ],
  "kernels_torchao": [
    "sha256": "f345c7c8eed10bf91594ab35f18d57e13453ee44b8518f07a1489247b456039e",
    "sha256" + debug_suffix: "8d3fd88c09d1652c20f9fd8d40534fa5786930f24665799d7ea46b8b0f13ae3a",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "608e978fdc6a180988b415ab22908b7aae8a8819e9ddf8652b823c7dbf3c0ba2",
    "sha256" + debug_suffix: "6928dc553f351b3ec75a6e37fb656331ad854a583139737a457bff5bd02c4d22",
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
