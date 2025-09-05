// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250905"
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
    "sha256": "3d69637bb57cf481aed164ecd2138d442202b81834bd9593d1fe74c0a02fbf34",
    "sha256" + debug_suffix: "ddd5729183b2a9a7e99e449c86d102c34496d0489b8119c40f492bf0963cb0a8",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "79434a31077eeab758fc0e91199959ec78a9321f4dfc350484adc5483ebed23d",
    "sha256" + debug_suffix: "f9614c267ac6f689047a1fa4e2de4b2a09401c8621c134742e09658f8d4a58e7",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7263ea27367b311abb6afca92ba60331da863bddb41eadbdd513e0745111e19d",
    "sha256" + debug_suffix: "f3b917db646b8b7675f52411225aace26f5852f62a67942fb30dc412f928f5d0",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "c41ccc9ac0d4560ae5e07cea54d447ecb20266c33ed54c4a96740d5ae13a14b0",
    "sha256" + debug_suffix: "4ebed8a84bc39be230861d47dea14bee81741c83bb349a9f3dd1cf74febb4129",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "ad8c86444cbd8c9d5c19ad177ed80cbf7918b38331e95dfd8f27f33d929bf244",
    "sha256" + debug_suffix: "1165b83af3a82d4c32c0baa94344a0e5f652be46de7c104ea21a81f68ff0c136",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "578e15df6e2ad5f018509d407c2993fd18f36ff72153ba488a3e4b11c573094d",
    "sha256" + debug_suffix: "d9f3d33151093be3820f97370b482385916fb24c83056708fd1279bffc8f7ab8",
  ],
  "kernels_optimized": [
    "sha256": "9afdf996631221a225a32383ce40742c0e04763cafe5ce8a293ae112e3b35fec",
    "sha256" + debug_suffix: "a08b96eeb86e5c066b8cc075fcc29fb3d82ce35f19e70d0afddebc5b49a180a2",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "92a36e55687d5901b2c745440904a1b1e8c29f094927bc99ce80985c3af5a992",
    "sha256" + debug_suffix: "497b304413350df9dce744d6668cc4464dfd540542b300590f09e3123fd194eb",
  ],
  "kernels_torchao": [
    "sha256": "a063b31b92368958183dd49bb0bf54bfd9744bec177b2e59ca457e0519374953",
    "sha256" + debug_suffix: "0ba405f2a0ccc6b7828feeaafbe0f7a43f1286c9b18c7e0d9d9c0585f39512ea",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "ef943322bdfcb8c2396bdf88d3d2002e58610d16aea7a220f4ba298dad7b76b3",
    "sha256" + debug_suffix: "d9ed94aac142be99a2d611dd92b3ff510c4f98bc97168efc9e55b95625b63b50",
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
