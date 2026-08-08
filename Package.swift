// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260808"
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
    "sha256": "aa78173d5d8027f67916f9a1b902f55725c7e844bd541c87458ba1c190c13daa",
    "sha256" + debug_suffix: "275e00150448c82f78f9152a3a792f6f2f04270143cfec8bb1f62d33342ffb8d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b9ea676da25308353dfe7b88bba6b76ce0563b9dc6854f5da54e361eb5a0a07e",
    "sha256" + debug_suffix: "860aefcc7ac679347bb11a57102f79435548d09980145af202df3dad5039ea77",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b0b8c2199c6298da93ef64c5bc8a49db9924a46b3370cb1dd121fa0b26f8b290",
    "sha256" + debug_suffix: "6130a32c4dcf138eea9d773469a73212607f44a907fa8c79ff3a664352a35f11",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5829200e1877d0b6cc3dce43a86f299d3db6d03675a78b4a14bb2174413b99a5",
    "sha256" + debug_suffix: "80fad4f109bd18755af179b9beb92d0830e41553c5a5d2ca6bdcfa4310a13bce",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "cb6f97b6472635b4705c50623fe602dcc02401a8f7f32229663afa91562e9692",
    "sha256" + debug_suffix: "5afac583a4409b57c09fa27f36c5ea7b4a7078eeb35755a7966db840876ab306",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "b978cad1f7932d4fb2fa17cb27223dea5a2c137645257ea6c3fc18ae6402f998",
    "sha256" + debug_suffix: "d590952de3ba4f92f4cebebef12c669053e26a09d2bb580d46bc05dd852ff749",
  ],
  "kernels_optimized": [
    "sha256": "ef9c31895ebfb1e2dd67834bae14ae11b4260639e348d8a7c0242e59ed1c03aa",
    "sha256" + debug_suffix: "7f3c19f71f374b2d151d4b90c3a841155e2c6eecc3fc673da942a9c372ca62f8",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "29c28889bee5c37dfba6cd7b99415f04b1e821be5431448a405111eac0840a9f",
    "sha256" + debug_suffix: "0715b656c7b1780887646a13165ebd51aad0bce78393c057d8da5f92840b14d4",
  ],
  "kernels_torchao": [
    "sha256": "7d510615563d8b25dcbf3fadda0ccb552958f64a2fb0b032441934f1eb02d2d5",
    "sha256" + debug_suffix: "71fd39a10394ecabdac8b404599f6ab3e585ba63b17373f17daae99465a47be6",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "5de6ff304509e9fa95467541ed8f2bd9752eb0e7befbfc352ff1a2d6e43f5196",
    "sha256" + debug_suffix: "3718698567bf0ff3e41cb5b65a889c72e4915e211c701e0063d61b5dfc4d2c6b",
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
