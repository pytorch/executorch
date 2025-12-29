// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251229"
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
    "sha256": "b3743d6d21eab2469c7647cd141e10b69d6cbad5f934da59dd22dbbd8536395d",
    "sha256" + debug_suffix: "01343bc17b8adc875baeb54e52351a1fc3bb675015aecb8fed023068584a545e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a31d93c5601d08ef79269759d9e63fa90ad82708eed79e1f64002497522daeb5",
    "sha256" + debug_suffix: "d641c30c19c3e52ce11da03fda3c54230d19b9582e974cf276ea97e96ba7c7f8",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "570ec005fb654b30f4587d060321a6349b7d0e7f26574c5ea5ed2d4eece35dce",
    "sha256" + debug_suffix: "a6b001655b62e3321aaa2b645b4cde0e3129f23ca3ca52a7a8ea23667882fbe5",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "1930ba253bffc62c10290256a10a26e27066bbfa0c12eca7af38f1e610ea3e5d",
    "sha256" + debug_suffix: "0127bfc2b08c6d1789bdc83687431dbee0e8a9468c3c71f5712516db18824b44",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "891c8ed93d50f6d4f0eb832d68ef81e050007db28e7996e601479241e271d98b",
    "sha256" + debug_suffix: "4f9742b82bd3bea25bd8833c83a7d4628bf58fb1c1653052d065b86e5a17474c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "1bd61bf5561dc3dee456efbdf5e8fd42da426e2eda068387f80dc3500d0a6cd4",
    "sha256" + debug_suffix: "da6a279efc1b7bbe7f0dd768b2b295f53f8869d122163dbb9797de639af2e6b1",
  ],
  "kernels_optimized": [
    "sha256": "9eb361380103a72edb75792b356fa0b80e810031d9c19689266269e9d506d107",
    "sha256" + debug_suffix: "cb4c4b2c5896699cd2d1b3c2961f4cca0419d0c2f49c52cb8f7d983db336db12",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b16b7e21026a68fc32a7b9e62af2538e4d8402d9de12331d4c11e88d36e1ae1f",
    "sha256" + debug_suffix: "1d934a6935d0ea143d29a8408d77eb4689b9f991e92473b42a65cb79dcb732bd",
  ],
  "kernels_torchao": [
    "sha256": "b1534842366c7ceeefcf1964a14ef30b1e8a613a18f52501c5c10b6b89a8cd8d",
    "sha256" + debug_suffix: "c4d55f02190dbfa65a524ab1cf7ae18985626adbc52451cab318edf839d1bfbc",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "3a2bde4aaf517e599b79b06d2a5dbb6b362308b213ac2a7e9c6eda578c8cb3cb",
    "sha256" + debug_suffix: "98623ada98d2d322d9f242c05d97b57fdf10c120ec74f081281d1e4dc40e9b9d",
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
