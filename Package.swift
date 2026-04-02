// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260402"
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
    "sha256": "719945687aad61f84d5983970436b39c4b448294d62885173439ad1b258e01c8",
    "sha256" + debug_suffix: "6614a3095b0b8b70dec483c62f1ec4ec40ab72c03a9a2ce435dd1b52b69970d4",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "64680e17bb6014e176583e7b735d05357a8e102a0d248fdf355bdf47aae1dd78",
    "sha256" + debug_suffix: "05b304a2a9ea2b4d07cf0490547789bd77aef74342e5f3c5ed8e6efbdcb3fe6f",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2c1b12f7f381f5b78772e811bb0a0440ef60a624c214b49eb231635ebc66c6a5",
    "sha256" + debug_suffix: "65bf926c71e9e81793d7794372c6d397787704fe6b5596ab81657afde3155d63",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f531c0cd3b1e84035148037c1889881fe3d5ac3c557c519bef4913559561b028",
    "sha256" + debug_suffix: "6fa39ffc7d3a4459ad7c37563ff3987c87200ee41e1ead595db3deddf0bd7cc6",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "2c7b85570a87c8ec728442444cbf17f9f00b0651c45182e5bd29d52b4dfb6603",
    "sha256" + debug_suffix: "a97cd8cbd58e31a7017e41107d8c076f727da1673247c00402fba648cd578469",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "0c7e2b02f8a6cd9a4188de6f5d9bf9bb632b9cea59210eb891e6b158a8e4f860",
    "sha256" + debug_suffix: "68c0b2b693da724a3194e2adc838d2e22ba7a7dcd716718a2f732d38adccda08",
  ],
  "kernels_optimized": [
    "sha256": "be5de79f1fd70050e38ec4bdab9eae94d048073c481bfae8d100b50719528caa",
    "sha256" + debug_suffix: "212f70778d313af4ee18363ec871c257d767cd2d02464427bc415eff0563b0cb",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "3526a4a35df0ca87b04fd0d808dc40b6f00005565cf9603d7ae2832f28b0842d",
    "sha256" + debug_suffix: "64b5747ec16dd48cebd232c3196473035ecb6b1efdd8152a4d9889d80d8ef936",
  ],
  "kernels_torchao": [
    "sha256": "7b6f961449427485a32e70601288bb78787884d5facd0ba35cefd614aed5e59e",
    "sha256" + debug_suffix: "2dc7ae01a3202b4d725f2bacdd7be078acdfbeea84bd99b3ec64e1ee9e5ac904",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "5ae028fa22b2e9645d14af8c55cba638bc7be1c91d461f210f879522b5b66f3f",
    "sha256" + debug_suffix: "f3f5c5a48a47cf159b2356230508a1f7fe4f627b86e452653b7d85a660c4f260",
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
