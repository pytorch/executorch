// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250919"
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
    "sha256": "32e5062dce33a72aa3692897d909a66f8645cd458dde160256dbb6a6778c8c15",
    "sha256" + debug_suffix: "30ff64ef4fb788c6b371ca0913759917f84817e2b4ba80d82058499877b1ab7b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8ffc79b1f24e5a60a0d7d7a50388e675447d767cdcf76ee0ba62301c3bc87ba4",
    "sha256" + debug_suffix: "7845758aa6418245d49f28a83d21659b43ee41c087d6cd0debff15722153f2c6",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "db038da7d4a9820684ee81a54b6543db4ea7c0d0dc7201ace43d250819e268b5",
    "sha256" + debug_suffix: "a2be717098e84c42f441b8195efa88de5b15808562fa7e0f87a59d103c05147d",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "db5afa4dc22c95732d51a2ebccb85ed320b07b900ade3e64e10ef97a3ae58b97",
    "sha256" + debug_suffix: "7e480fc0cdd6816f3cb6a1529aa5d264dd87d29372208b3ed5de43ad79eee768",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "000f78a182869bb457f9a01058d53a6e23195283f780d6b01aa8f665ca49d6a1",
    "sha256" + debug_suffix: "8420f6ce4b05dcc951a06ad65c1d3351f66190d769608fdaca0789211cee8764",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "4ea38fc5dad46c9ad80a512b86981b6f88ec00b88c8c21560e1254b54569d272",
    "sha256" + debug_suffix: "9910d423532c97ba10cd5bf9953a37588eedc21666488d1b786d0d72baf926a0",
  ],
  "kernels_optimized": [
    "sha256": "5f6eb70ed43088cbfe26c4f4af7d6586500b6c334e11552b0a02b43c3090ed68",
    "sha256" + debug_suffix: "193cbd2921e79f0d512a58bdd07ae91a91f3516974d2c472bcd9e16a8c09ac88",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "7702d104274c491344613632bbf61b87baf76bcc26a458a74bdfdc1c158df112",
    "sha256" + debug_suffix: "d7cf6a60cbd269e0dab0873baf4d3fbd3e90fcc11e718c21af218742f75818f2",
  ],
  "kernels_torchao": [
    "sha256": "6d87fee35ce7c37b0aa00b793bd4458fa033a2be53d5aa2ba0d5376d5d02204c",
    "sha256" + debug_suffix: "bad71659f1c89249217edec01b098fa00368b44895371552c42a6c786424b682",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "71e5e43042a3a786f3a443989cd50310aded81d2481bbe04ff1f907ce19db287",
    "sha256" + debug_suffix: "fe7866f0e07ca13339c2a627c9f9727393a9cc247c48a809eaf7799c483398ea",
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
