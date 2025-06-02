// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250602"
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
    "sha256": "98a1380e3cc8245a23f3fddaf3cdade3acad065c701207f4feb10ccef9db17c5",
    "sha256" + debug_suffix: "be5e7e766ed1732892a389485a349d97ded2501f7731ef66666df14f7b2d2cde",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "63bfef926560af22b0a07c436f8161611ce6fb7aae600c5d4a832478c77b6d6e",
    "sha256" + debug_suffix: "c69d7f4e626e329dbc8e1945c1bdc7df8ac54935301af332e31e4811f880ee96",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "909cd4135a2d46b778de85759996ddb956f87991791cba79587459ff28c665df",
    "sha256" + debug_suffix: "9d91b4a860230a065072f98e12b81736c9e5d14c3308edf663d675b176b5636e",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "840b5471a0f5a540c0cc8c61df8643650bebf4ed13d0de44197435b7aef3e861",
    "sha256" + debug_suffix: "6c394085ddbaed6f4cf0827da6ac5af032af73414a17e416ba4d6014e02cb568",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "c9338cc09c93775ff1b92be988c61081d00fac82dc92c501cfcab0b0d5b1c299",
    "sha256" + debug_suffix: "d010757b77b27dc269e88ac4f0ae51a4e608569aa6d1202b87af9365e214d555",
  ],
  "kernels_optimized": [
    "sha256": "837f2e4d8ad4bd3618304570cfc0e70100737b9c2dd86b1a6db831e108604e9b",
    "sha256" + debug_suffix: "fc71954e8a96f779a24842fc1d42aa7bc933713a1ac9f97602533f694fdb7ef7",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b733268772bdf827e05e927a22a7337ce0d9d25a4268f013ea9abefa213e06f5",
    "sha256" + debug_suffix: "30b5bb7bff667e63dfcd17f3914aa596577d973dddfc07421a3fc8711a9eb9fd",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "d990eeada71281cec876f6b6073bcf54cd288ee9d809d748f2edb908f5a44960",
    "sha256" + debug_suffix: "d3d3b4320b2eea9502b8fd3ccd2d2c70588a198c6476781a53b40c44c5f3933c",
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
