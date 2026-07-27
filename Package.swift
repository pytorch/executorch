// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260727"
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
    "sha256": "b3f6767aa17ed78bfe4ef814e97e4d2488b3f8ff1a4ffe82d62015166415c7ba",
    "sha256" + debug_suffix: "155a01df66c0e65b62b78010c29e36288f72a953d51a61da0ea4c04ad88c1729",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "13bdf04366281b2289a09552c51a8d8f61c27aed93a20dc754cd5fceb76f1dd1",
    "sha256" + debug_suffix: "e2afcd6f852e4c44d484f5533f76b01bb208b4775ee7e7796f59048a4b42651f",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "da0e033a2c88f41ec5587b7be155ec496d43b96d90c567f58c8dd5300fe92ddc",
    "sha256" + debug_suffix: "7afe3b7c7c8cc6d41002581d32f704ce1ec4be6109465201937c7d9dd8a0ed21",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "63b7bdc7d4598bb51eb856c91f99c4f86d6e1dd5821476cb120e0618355ec5d9",
    "sha256" + debug_suffix: "4a2272af6803724136ddbc957eab137122dbd8c5bf62cef80980338b927dfc6b",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "4e7fa2bd9f7d22bf29501de354ad52e46d0a5cfd2171caeb563c660235a6b61b",
    "sha256" + debug_suffix: "b0bf5003f081cb2734ca27b8e219f0b2a7db4e649d89f983ec505dc7ba76ca39",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "b152efc3b7248563be15aa7c4323daa9520f4c033f2ae0e8aa9f56c2b3668f5b",
    "sha256" + debug_suffix: "6266b6be2844e5f15cc67cc6156350838bd1e03950903decdbfe700ae89f0fc7",
  ],
  "kernels_optimized": [
    "sha256": "39f3a255432793b11998e1ec91dd63380f3f78634fa0320434d4fd1aebad5256",
    "sha256" + debug_suffix: "d10b2f5b7e18ca36c562168bcfacdc024398d5e6bd6650ad478afdb7fbe090b9",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "04f24f7ee78b33f045a972524fd97947f22ec438b68b636081e3327f30d773bd",
    "sha256" + debug_suffix: "57907b267757b8fb76342c607b6a610afebcbc1b2c523b1588e9a62fb776d337",
  ],
  "kernels_torchao": [
    "sha256": "780c3575ac0ec3e9ce23ed33dbbba819b2e442b867a53694a61089147056fd27",
    "sha256" + debug_suffix: "8fd2688c202fe865fe5b26c6627bf9035b846a2ec2eb63805ef327e87c3e414c",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a87983b88922e5d2f89fcd915bfc88f5bff5d26529938ea2910ad151c41ed38b",
    "sha256" + debug_suffix: "66582b1ec039b1feca6c3e0d2c69dd7104ec9fa5aecaef6b3c93b5c3df067770",
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
