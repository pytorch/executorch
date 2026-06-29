// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260629"
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
    "sha256": "746227b70e25105ea714b74eba3b85a019b727ca0b2417e8004f3a0a9916c8da",
    "sha256" + debug_suffix: "f4b3288c8e40cb2c8ec76a1fa09b4d9003cbd89bb4c531352f6827e99b985d33",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0159fc8bff8b9dd4943f004bc736f76b9dbda215434f05b52b0194a5e66f39db",
    "sha256" + debug_suffix: "724a907324b4b498354d31e69b6f5f0e70d9efa055d856d27384dfe4dacabe0f",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "40724f27190fccef5b3bea8575720a0ff4f037dc29fa442fd4ae77f3ed0ac46b",
    "sha256" + debug_suffix: "f2dcdf1d7446c96439d44311f78e681b62733a28120ae7afdbb41053118e472b",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "740f16bfc4d90fe0b84856c6bb1956170f9fbaf595b0726510e6725f06b7aeb1",
    "sha256" + debug_suffix: "7503d6b0b7fde45c021e0d90175d1910524690fc52be11bd234aa7aca4822cc3",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "3319e74e0736fa48236327f491ae990c0ad3bb73a6c804dd70a12aa01ff611a3",
    "sha256" + debug_suffix: "0c6e1adb10831520cb24029bc91d9d00bcc3cf9e85e88e70eb2f0a97cef4998c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "5d830787aa8665686348f05b7ee3b3eb4c55ed7b7837b318ec92c2f404641dfc",
    "sha256" + debug_suffix: "d2d59386b583e570b4b2b296779f4573b11dc0e2d9ee933fa0e475f8b71b64d9",
  ],
  "kernels_optimized": [
    "sha256": "4697176637f6de6267e90497859c3d7f95a951fb17ee0de675a589ac9bb08ae1",
    "sha256" + debug_suffix: "80d0ee9e8d8b8206cc834686dd33a7f4affc8d44533a8ce74a2959c8f239eb4d",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b84f05572dc0d4a2e3ea0530bb5f622bcf4e17c9674ff633e22fa2ee21627291",
    "sha256" + debug_suffix: "ff0622a8b95d8039d19d904897ce24c18c53acd4ac99d6cffe1c035f5e0e8c25",
  ],
  "kernels_torchao": [
    "sha256": "9d472aadb67c17082103a4929d3745612f5b28b51a8d512721f3c4d5218a5dcc",
    "sha256" + debug_suffix: "1d2d47005e7b2d65ab130e5ad1755ac12542d3d69c63f92735143c17ce331f2c",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "734a37cb53ddfe98f94f8c8ab42c1b9e72633ed21e63386e6a45698cb834c647",
    "sha256" + debug_suffix: "9cb9f4b047ece2490a963c9d489f3e80e9a0ed118295e9a1d70041d83f2e4add",
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
