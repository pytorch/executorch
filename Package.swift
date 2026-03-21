// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260321"
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
    "sha256": "717d7864714158c644dcdfd3d824b522b922f2c81cf9c8f3a17df7cc25eabc39",
    "sha256" + debug_suffix: "15eb6ef04fcac2765800a6aff96f6ece2271aac7e70957fb07861fd255cf4d55",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4484c0b136238a1ac8f84076dc2ed022aef3e9f458b29c481080aa8390f006ca",
    "sha256" + debug_suffix: "6cfd89fb8ac6d3220433013002bbda71a5df4fa76e65265c819c54ed287c228d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "65ab1653b45c93e0367ba2c2e639f064ef51eab850c2afa322c2debd52366d24",
    "sha256" + debug_suffix: "0dfa2cdb0709a4ffa25f680670528e8115822cca7b8dec733592de2860c909ea",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "48324115023ef6daa41d171d276e7a4c4636d9077990b22217bdfe0ccb68d5ea",
    "sha256" + debug_suffix: "46a36907834a756f052d913a210d25177ba477ab1984f8a1d915d8ddfb05d55a",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "c904d95dff75644e9045ccb446d0f27465dda98a6ed0a4ebd9766528ab63e280",
    "sha256" + debug_suffix: "d68703534f5fb7087879d14b1d4f9773c5b0c66625d8c453c683f61294799bfa",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c510d52ddc77fa1a0124dc9e7baaf93862750e3a6893cbc1f13197ba802656df",
    "sha256" + debug_suffix: "b87234d81fa4cb4ec9dfbe7fab42acff4a3259af43613aca7aefca13b0b38e6b",
  ],
  "kernels_optimized": [
    "sha256": "52404cba4496c9ad4dd11060aa252b55e6de74ddc5a9fab3add3ac6d2e8ef0bd",
    "sha256" + debug_suffix: "396ba7a85a50cdf08ae365cd9dbdee41a12a3b4672259b6102cb4508670cac6a",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "0cbfd9b358041b61e3209d0ecb612df4698d3292d127ebba21e6abb4bb6ddf7b",
    "sha256" + debug_suffix: "7482bd6d323d0638ee5b473043c69ca5a8c9428ddc7880b456f170aa32ae448b",
  ],
  "kernels_torchao": [
    "sha256": "fcb3e74e892da8da8a519be0013dc33d9a7a3290baf19ed9ffee47c4d7e9ab0b",
    "sha256" + debug_suffix: "94fc8e07564a8c811879b38de7f89ffa125548781ec3ba5f79f5f41ebabc328d",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e3eda92ec6afd651b89eec2dc3272be2dab4afe4eafb37eb314b8e875801cea2",
    "sha256" + debug_suffix: "27fab0398043025232b2f55e75767b91c186b248c8b660b5ac4ae498c7a08e1b",
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
