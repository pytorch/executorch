// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260101"
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
    "sha256": "6b790b833a354312e552bc6dc129e21c9dd35143915c1790e2fbc8a0aa266283",
    "sha256" + debug_suffix: "bf3d36bf8d504a1eabde9f64ec669bb397a05233aa44ef3bde81dd3d3058a781",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a7ec25766cacdbf551ef3042386d3204866848eac232ed157df96bec2d82a06a",
    "sha256" + debug_suffix: "f9f258f3c81596ce1cb48c39f2f93bb2d59d3225abdeca5ecf5fc2aa0ec30ff8",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "113ce98f03bef4d977872f9dd981a4110c682d470484a914e4cc6fb3dadd0d85",
    "sha256" + debug_suffix: "8ec2035c107ba652c283905377427d73cbb6186e34d640fea16fb21b692ad510",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5667f590216119ac566862dab60953a775de8feaa83d17ff43f85eb1b3636d05",
    "sha256" + debug_suffix: "9b6853914850949c718a1d0ef03afabceb0307f03d0908250f53433dfb2c47f3",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "66e1a91294ca1f310fc1bd4aab045762c27b191c3236fb31f72bf5eafab2566f",
    "sha256" + debug_suffix: "33cb52a1363e2edb3688e7a5aa7f932d849c6a534a52f09a45fced94299f1a5c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "119bc20c978ceeece65a39707b33ccd2e02eb50a8302b7f330fe5785e46bd360",
    "sha256" + debug_suffix: "0ec2d8c81773d45ccb8661d2bbd67b1177fab331d57813e8d259d65042db9e13",
  ],
  "kernels_optimized": [
    "sha256": "68107796802e146e62883ad8d38720e58633c4fc0fa0b29f58b7407f0769e434",
    "sha256" + debug_suffix: "bb37d371bbb5b2fa6885499e583470560f9d6144ae512c9e1bf4157cc0e82313",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a051fe8819309bf839206a2f5c1d7104824b96f910d2d6f02f5c010b984064bc",
    "sha256" + debug_suffix: "35c822c10764dfe7541fa086fa4456f2f18c5bbbc2cf7193758f8a99ea616ea3",
  ],
  "kernels_torchao": [
    "sha256": "d2fef87a98bd06b79817815f367896bf9b3d055db65dae62157b638aff2e5713",
    "sha256" + debug_suffix: "7c4cde3bb273e1bfc96792aaf2d5bbe9b39262a2df67cfabbe7b7167f99eac30",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "d97a0360c7a7986c52a1973f6c8187527425304996d48511d87f355645812c5b",
    "sha256" + debug_suffix: "4b62999f8a5587c4b2c5a12a5e3d187650cc8c6d384a53c07cbbb9f32c765a93",
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
