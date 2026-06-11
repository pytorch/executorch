// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260611"
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
    "sha256": "5bffd8dead9b1ab05eac0987d679c681b906f3eab2c5c558fe79e63515d14a86",
    "sha256" + debug_suffix: "6454ff60aa8d856b242f070a1f02f96b53d55adcbb196121845d3c5d8a3b7ad7",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4255326e59effff9ad8acfd45b351d016cda44c027bb6070b1ea85566319dd18",
    "sha256" + debug_suffix: "c9363c40ca448a9de4101f48a3324461575ecd0274b345ccb2a8d4106a4618b1",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "95c245a55ad11680509d4a7121af62b78ebdbecd172fbf549f72b6abab4ac9ed",
    "sha256" + debug_suffix: "358d52cf558436c22aa709fbf0b21381d171e81201144be77053d2bec5eadffd",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "232c5d7bf2ab66261f82cd9941efed5267547d9993c9c6b1ab447c2c4f5f14f5",
    "sha256" + debug_suffix: "f7a12862ef82c48bce29ce9bac493af692ec49b32f6a527e53a7e6295d1852ce",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a54e3fdc0579b08311b4c9d79eea45a55d12851f8c60246bc670aafb5247b54e",
    "sha256" + debug_suffix: "3e45916d762a1c0b75c43d722ab59e75c5442e3267caec3ea446bb18c24a54f6",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "13878682ae179507d9f7fa420e22fc82f38dbf0d3c1d8d93a99576ed6676c969",
    "sha256" + debug_suffix: "a95b13a6a2feb528b1bb3ac977d96437e215b3624213318b284fd3367feacb24",
  ],
  "kernels_optimized": [
    "sha256": "67bdb275b9ef21c4b1ee3a1cb321c51a85a7912c3263e3549d79311706e3c13d",
    "sha256" + debug_suffix: "e4489f3856df71233c2c8e788c44ad9254b560b886810d463f6b1d8101450c3a",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f99e1a3da31a1b65d52f48d94f5e66f67808abfbe14f9b004bb145f94b56bd36",
    "sha256" + debug_suffix: "daa125297df61b2f31d7e95ad9adc3b6eec8a298d07fa54cd11169763a99cab9",
  ],
  "kernels_torchao": [
    "sha256": "8d17d2e4ab582046e4b70e27aa03e76cdb80ff77393639126032960101ed27a6",
    "sha256" + debug_suffix: "82e1607aecbbfd76d7e5049681cb0fc838117cf1d49eab4da4368d33df9d21b9",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "ffbc31425390538aacead0105e10db4d7680c0a32154de873de9734055c5900e",
    "sha256" + debug_suffix: "48bf473d1fbf43f6635968af3fb20250da02d036fe0aa9d98ab0ef06b6aad37f",
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
