// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260323"
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
    "sha256": "a27e1730bd2f1c10092c12fdf9c63b070f7843ad7eb150a8fa2925ff492ee697",
    "sha256" + debug_suffix: "77faaa6412f833d45b762c3b330a9e287070656dfb1cd7ca444e32c4a43d462e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "540574d98a3b90e2f147efcf036664ab09a59ff4bac894f6cfc6afd6f15b37be",
    "sha256" + debug_suffix: "71b740fa31cfde86ad7bf32368577e86f05089ce9a2b292f5cf10009533568e9",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "03dd8da01e567f05b9bfb3e12857df97a93f5d3d0c0abf5df72a388518a09ef2",
    "sha256" + debug_suffix: "77b291f4db77fe737ac3c7706f4cc3b9f428a6be55d5df758467311b622c4346",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "eab108b75ed76bc5baf5884f90eaf3723cbb3cb6fa8cbb1b3edbd609e35a776d",
    "sha256" + debug_suffix: "f7a81960b193f63f04967865791305de9b63d1d769dd4ecc653520b5615d58da",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "011b8b222b1e5343e96fca7f3b3e1dd7cd9b8fe39a7f260e1caf70aa50ec0c5f",
    "sha256" + debug_suffix: "142e53a3e10faed4ffed62b7e8c03e185f0701bcdbce261f4c017a6cd80ae916",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "d26e64d2f85645e96a3b24afc86bff05d8f0ea515d500c617b675f7cdc12a55d",
    "sha256" + debug_suffix: "fcd5a3368b9f8fb69236c9864cf8831300a32fe98f42c6a1e46960a7f5cc9595",
  ],
  "kernels_optimized": [
    "sha256": "e435013b1d6eaf1807e0bc32f7f75332e4943f26d87e08b60db1fc734669ec83",
    "sha256" + debug_suffix: "6bf7b5fe3bafb4e192082cd9428a2cd12a61df92235312554aacf77a9ecb4994",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "3d2267fc3f233e8271aa4a8a048367ca958bd90a6a6e4b8989dfd3e97d3164e7",
    "sha256" + debug_suffix: "d771874ae6a34e10d5b89da70dc04a0bb161e465fc7f7c010df88cf8dda392f4",
  ],
  "kernels_torchao": [
    "sha256": "76b9d84a0f386419f8ede105763b5641aa7a06a052b4883c35aa0b757fdec77a",
    "sha256" + debug_suffix: "c3db8d49c113be1b3d4b76851fb84be157a60ec0ea43fab3fd223df7ed6ead33",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "8943f6d83434a02f837055bfff2f39afc58526eb0cc2ee52a98894df1787cccc",
    "sha256" + debug_suffix: "72a5a693b8fef488e41f81673a2fb22e4b09b790da7a3e96369663e83ff817ed",
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
