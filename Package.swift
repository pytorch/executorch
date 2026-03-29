// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260329"
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
    "sha256": "7bb2d8d4080d6e268d4d67e6c06605b43bdcdf42235e296d72e49deab83152a1",
    "sha256" + debug_suffix: "998768f5342acb5fcb900c2af8d65f4d02c54448237020ae7d642e51582973d0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a6948b2d93fad1f378d17164df8540d047ac6b5e86daf98ce0405421e8d5207d",
    "sha256" + debug_suffix: "d9f8e0b3dc284547054b87ed7f50cc69d21d02de872822d3e6f50918d4afebfb",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "cf745f694e40bebb2175d2949d6446a5fd97798149859c96ad3ce92d66a08e51",
    "sha256" + debug_suffix: "65ff6089f203b7212dc56752384e06abb574d69e27e23031497dbc7e10879149",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5a6d904297945723c7e6c0503b46bcae290dae4efbe48103b63d490b2f77deda",
    "sha256" + debug_suffix: "2e9949c63e2fa46d27d05f7ae9be736bc417a46c120a25571975c4e8296b1c5d",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "8858d570c41666548813b3710c7f8e5ef12936447f5d7247a54e0ab9eecb931c",
    "sha256" + debug_suffix: "0ab71e396c274457da4237f859de2fdc3aa670464b540c1a1f5cff4e9b1a75e6",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "71fe7769aab284ff0308112c672562eb605098cc89b7dc0e9fbac7c3dfcef5cb",
    "sha256" + debug_suffix: "70d8980055c45bc4b25f9325f580a05fcf1c88471997974c0ec197e20ac5f5f5",
  ],
  "kernels_optimized": [
    "sha256": "efc48e54de757e8d260d324748d008e5188aa936ac123ab7e7fd9c527d2c6782",
    "sha256" + debug_suffix: "9e7adfa3de50d4d5f9fd4de724528103d1583a669a0d941fee0077605694c47e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "8a2a07619b1d6e1a6b511d7b4e27fc151bae64e3d4199125fc8281dd192d44a7",
    "sha256" + debug_suffix: "278b783057153fd1b1c28305cfa113e4664fa8505c9370dac036885298b0e104",
  ],
  "kernels_torchao": [
    "sha256": "9120076f30406318ccf8fdf4fd6f5e7eae66c1d03059f586630e1da6654ad777",
    "sha256" + debug_suffix: "b8831a683b46b169005796cab771b57ec65e15b0a7e08b4d855d0d0ddbc5eafc",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "29f68d167c4bbdbb0f4cbef9052828ba3472c25b05160803f4bf2c2c3114e180",
    "sha256" + debug_suffix: "152aba740b9807b65bbe4f3c047fc3f0efaf4cbbba78063990b50ccd005c4919",
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
