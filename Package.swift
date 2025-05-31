// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250531"
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
    "sha256": "882291697441a6f7689f1c438960767646419dd1ecb59a91cf30b6a69c44a183",
    "sha256" + debug_suffix: "f4c6f72497ad5d09b75cef6c2c6b7b09396e4910dc0941bc8bbe7b4da777c4a1",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "784cbb5efd0162343d809fbfb7cc334b3446742be82bc391f47319f40a2c3a6e",
    "sha256" + debug_suffix: "4e94e515439a3a4e86be995a61621a7f1ee982274d530793290b71f433b8d0b2",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1edb167b470f5923aabcf611d87f4fc058f446b02415a4d99ee874e5d4e7ec8b",
    "sha256" + debug_suffix: "00f9aa26749770a1f50521af51ba2f7ad2a1d6b9abf43916b303f886de2974c1",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "4327976e53f0daf06e95981afc8f387363699952592435457827daa78fd53f47",
    "sha256" + debug_suffix: "0fc2ba7d4c300f617c7fe39bb52e7a37bb9fb15f387f62a4078e28c2ab81c6e3",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "15eb1c90f799787a22ff3f5891fdea358cae9ac3570e9c2cc4674a478d96f4de",
    "sha256" + debug_suffix: "901eaa863df434ab40e7840e0d99dd88ae624b90ade8947b5ab6b0cc7d1d9e09",
  ],
  "kernels_optimized": [
    "sha256": "0a3e4c4cb2fb642f008216af92105a20b566b51a3084f068bf7cb4c151895a05",
    "sha256" + debug_suffix: "0c28fe3c791714c8a5a5dfa2fa6c515cc2f2934b2e1ebd6c83dbe1355e21f3c9",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a2bae1b82bac671173b716f80bdc554b3ab2e1891cfeec5cab31af53d88d2afd",
    "sha256" + debug_suffix: "04425265eea2c13271b0b9243092634b93090b1efd58bfeaeec073a65e03cf2b",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "9d2fc9d96f069d2b68321f0be387064fa160c3a0da5af20c18655a3d715b1f92",
    "sha256" + debug_suffix: "7a187c755ad0f35acab0b3a18c906480cc0809543377868927133ce6be4efae2",
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
