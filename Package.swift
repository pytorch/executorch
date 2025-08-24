// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250824"
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
    "sha256": "e8a7aa414655c159f2fb86e2f59440f06ba4414f2655791bbe66c1fee8bf13e6",
    "sha256" + debug_suffix: "ad2a201162dac07c7d70aa320783dca52aa8e1b99a47453831a72195e3edaae1",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ccbadc0abc1573ac6016d661ae39d4cb464836ab5cd91ded38be5471d286cdea",
    "sha256" + debug_suffix: "e73faa05d487bfa578fcd12e30b56081f679c2ca784187ed50ae26bb7eae993e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8ecdffd663525f452da9a283195a01b05b5321323b153cc0a40a897ee79cb2a5",
    "sha256" + debug_suffix: "54f96ba5f0f38c0472c3d572f6c04636d3c94600627a2588f3639f8b1a780878",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5d29b73d1ba22e7c2fbc15fcd53555ec4e97eaceee17f8ca5c7fc4ba1ec09ca9",
    "sha256" + debug_suffix: "1924dd3b43ea0766434d5547c6ad78bdef461273c706bea0d81c2fda33605300",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "180d904a2b30c8504a284cc6dfb39cbc3de3695816c4f6d669ea29f09371b9c2",
    "sha256" + debug_suffix: "5a91cf58eaaf0431f64160dd7a664bbbc410f5758144438f17c1c69626c4ec44",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c03c3b3543b3bea95d46663e53f9ba0f567e180c8be38c879080a9b48455bc0b",
    "sha256" + debug_suffix: "17411cf0f577fcb404f0dd59ee082a5822c17dec3aa45ad3facd198678468526",
  ],
  "kernels_optimized": [
    "sha256": "94606178d441faf4705b3e6d0a8ab3dc4cdd4f0895edb8a7d99c5397acae0a34",
    "sha256" + debug_suffix: "aa27a645452c9305cf66f6b82db8bac9ef17d76122b22495d2b5d71fb6cbf247",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "cdd714853d88df4d52ac1124b06d895aa4190d25575e699f0c61e0c3408ee7df",
    "sha256" + debug_suffix: "e8e5b338231b26279e66917983c20b2c6c2376f9e9f026b5fe44e0015b02daa8",
  ],
  "kernels_torchao": [
    "sha256": "__SHA256_kernels_torchao__",
    "sha256" + debug_suffix: "__SHA256_kernels_torchao_debug__",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "302134f1fc1c6ce9eb429371b8e96cc1eb6e47096c1d5218f5fb9b61ac4b2207",
    "sha256" + debug_suffix: "bd46546dbec3aa04334f0970f228a137830a9f6f59748c482effa8222b0ae5a6",
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
