// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0"
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
    "sha256": "536ae53e60265157d8bf058bd87c7cd3c2df2bf077172a9d64042db7ef1607ed",
    "sha256" + debug_suffix: "373c5bef61f62db3b23509639ec8121ea9b770a51b96a763efb3484988a4a148",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "221c56a7a6c6a5d02b622cbe3cb345c9edb556436050c3545ba03ecb438e85a7",
    "sha256" + debug_suffix: "13eb3a7fd74daeb1fdbc95093cefdd2fa3133ec0d79f2ac662dea151d0b79bc4",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e676641b8ce773dbafc575d68626c7f710fdc898b5bc5749a77270bc3fae2bd1",
    "sha256" + debug_suffix: "2ddac261847367abdea4743732fa634ccbe63c0d21ad402d0ff8ba7528eb7fec",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "14ee5df4b64984a4b834cfd92234b96222f483b4f19f6d193b45301d8c8b4527",
    "sha256" + debug_suffix: "f98a1140545dd197e6baf63e9dae73400189a77b5e1693e089cf634ab71f930b",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "24e98435a7c19d52a5598094e16257bac801a8de243a9aea3ed62caea02a6196",
    "sha256" + debug_suffix: "95de335a106bd1e7277d69c5fa4681711b281a6170de375fad294ad4f071405d",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c031210f05d4068b9d7ff44042b2b7cd260f2cb139e92d24b496970a9453315d",
    "sha256" + debug_suffix: "92a9182377a953f286e804c99f6e4a696e85acddcb35ca1e382d040b0d40d260",
  ],
  "kernels_optimized": [
    "sha256": "d4a7252a82ff744f43861cde69202ec250ebab67fc11101ba791f5c8f0fb8735",
    "sha256" + debug_suffix: "d6f568cade7102bde28c3aadb6814d34cdf2867b023bae7acd18490b708ebee1",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "65c858eb3bed40eb6bb853ee065ad91117abdbcbf6fdba6d5177980e72f2f198",
    "sha256" + debug_suffix: "e200df36c58c5bc25b051644c71a670f0d1693c43bd2287458c182a3dbbd8aba",
  ],
  "kernels_torchao": [
    "sha256": "619e7fe9e74a329c63e8b0bfd5bb6b4715e670e25a6d2a87d7a0aca3cd1eb1e5",
    "sha256" + debug_suffix: "33a2d23edebd52c476d6ae8e449b8e0d36f698ddd0f1d0910b83ebed5caf5104",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "bff9ceee8e022498160ad733abbaa4ab8e1d87bf636ca46ce5880cb17b9661de",
    "sha256" + debug_suffix: "2a50b8f301e3cfa9a6ccc88048e58073130f41fed75b721dfd2cd48ab45d3154",
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
