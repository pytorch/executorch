// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250829"
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
    "sha256": "f636dc157d9a52f3457753973d1e9e0c099603451217d263727dc6771885413b",
    "sha256" + debug_suffix: "734289804b1479b0557ba9f9e6f8a3d75cf4180c2c6b667e746d76ce2b84072b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "58c75e324e98c70ff4f6ccbe41de56cdcfadeeb5866bde707f3cc46715e2ff10",
    "sha256" + debug_suffix: "c48137989b9c7a25cf8396d904d17b239eb50a709aa973450f190ba6f6919ce2",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7e8c3009a61837b04b24b4b809ba32999a1ac50254ca5fa9c6f164db742d00e5",
    "sha256" + debug_suffix: "280d17fe89c17c6aa8de8ecbabfe6dfe9830d18b7439efa5f05f99e64488318d",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5036c3321975a78b37c2ffbdc5ddd97c0ae902b527fe3ab1a8560f4a09dd2a13",
    "sha256" + debug_suffix: "e1ccae8de9cb1dbba64201d831051a0c565926853c5faf70c08e423f1b862c24",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a34e9cc1721e93b2a4ac365ee01124855127608c898d006ad00dc30e73d3d9a1",
    "sha256" + debug_suffix: "b40c686cd6fd2aaddbb3bd6d998e3cc67cb244db7bf3b4617e6eece08a5e4eb6",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "ab8ff403ff75f632f59a2285ed4027f02c2505e32756109b7fb1223cd9e212bd",
    "sha256" + debug_suffix: "19c926de43fdae79f98acff87168bc0c83c9f73ae38eba1ba2734a9e41aa1e7f",
  ],
  "kernels_optimized": [
    "sha256": "f982c3a6dc35da69d219f1d42208523c8e446e4eab75253d78431f9f95ce4915",
    "sha256" + debug_suffix: "7c6880e8a501cae5a6ae8d2f67569d18272ba6506ed810401442a4923a70891e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "427dae7fa45755724397eb46d358db486a3a37621b1709eb1fdc48316753726b",
    "sha256" + debug_suffix: "592a023ad412c3f6817ab32852b8705bf7f9311085f33fe913582a4e0e304442",
  ],
  "kernels_torchao": [
    "sha256": "d9d3e290496f4ab4a35ed0ea5775cf660863a62a57e68c7b7cf75f2ccd120ca5",
    "sha256" + debug_suffix: "acb48e9a7aece05aec4aea9b75b8288875ca4fc3b822ff7fd90b9609bfb1d777",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "18c11acfd1b95bd0f0ca515b17df21d1f2e6ce8adb5a5d734eaf0d7f102425f6",
    "sha256" + debug_suffix: "b07720d5150cfdcf2ee60b57c557b6fe8af27486b6d3ad9f915f3c732e0ee714",
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
