// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260220"
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
    "sha256": "9dfb4ac4caa0971ffba294b1b0ff5f34d7d96b330fba4e636fff1b87d5fa5fec",
    "sha256" + debug_suffix: "ed5c3382d738798f1af9319ba133a8be38cb493c22cd1215da6754aae236541d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0353602a4e7ccbcfeb43f99243e1a96387c3f84ff9d1140289601f7f09b16ec7",
    "sha256" + debug_suffix: "d9d1f3ab5aeb16d979acbef443a6014db3b7bb69a5bc405f14f3c7bb04dc8fe8",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "21ecf148149285caaf9f2ddccbd7b3f85f79ffdfe50663f65281d9c47b456187",
    "sha256" + debug_suffix: "2527745c9a38026a3e3f5fc2733c629fe479ba34f5fea819bcd248fc19e32513",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "696c70fe126be43fd6a9851dfc0b722af20b4d6d5eb4b2d667eacdb47549ecd5",
    "sha256" + debug_suffix: "529bdc901321dccdda0709bdd3353dc920212636d1f4e41570df2b3b650d830a",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "3a3993a1b10b15713ecb634695a0bb113d432fbca4890cb37339006d26234b7d",
    "sha256" + debug_suffix: "1584086670d71bb0e738b80494bf0969fedf910a6a5b22728e849584e3a531b0",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "eb4b9b530c90005291b9d4de00c9e6d173bee796a959a5ad9f6bb615cdb108ff",
    "sha256" + debug_suffix: "addbc25ef1345145fd45cc83bbb05b8f1e6e0e7af90765a80765b91af7fb8b30",
  ],
  "kernels_optimized": [
    "sha256": "24eec690075293d2e622223fa7852d04e300254b2c2e43cf74c484b6aec66125",
    "sha256" + debug_suffix: "3be95b3459361129a3e749e23497c1eeb5f7e4189d257ec9f10589727a683b55",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d4c61f55ce9f854fc3d5f442acdec430cf0b8a84f54a197caf85981c8da067f3",
    "sha256" + debug_suffix: "993059163e933c407041d2b46f1e46bebd587ffb544d1ff705a9459f89f5da01",
  ],
  "kernels_torchao": [
    "sha256": "aecd7d881fa4bb0d28ba461aa1079de420ef110043001d92688b5900e6fd17dd",
    "sha256" + debug_suffix: "12857753b3796e4ad4960b9d938e85889e38d392ac7be4616e1cf4bfa7f0b87f",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "bd671984b0ab27ea0dd79ff3680e46e03f13fde3b8f1f08ef101fbae0716005f",
    "sha256" + debug_suffix: "65de3fd9c4113233d5cb67911e1ca2c58ba2e5d993a3e6a6223016eb972e820f",
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
