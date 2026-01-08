// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260108"
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
    "sha256": "ae6e66e8bc027cdb73102b899b8defdffbc09a7f85d49868a9d60e58e2d48640",
    "sha256" + debug_suffix: "2d65d674f01d5393e2e7b2d44d9a460174f2f762328f201e9ca9e0b2ebe860b9",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b6bda370a00e4d55a311c0f4aef92c95fc6f5279a9ca8f5cf529348e80684944",
    "sha256" + debug_suffix: "a4e9f12eb8af9d1c5defbf8b5743e171075f03f0fb1cdfa2dde7724beb423ed5",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4e31c40a839fde3159db576611b8067526e0c9fa2ac3ad9f0fb2f24269741779",
    "sha256" + debug_suffix: "4cfe08a3d4a353fc668a864fd1d4aa6a3bf5a9a0bfd3f965065ab939f60cae11",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5623ad3b2b7c54dc35adfd2395c650100dd747ee5828451324bb89fa1e451d50",
    "sha256" + debug_suffix: "1bbd7e73bdf66540706beca0f722d33b0bb0c02aa2841ddef39a34953266777e",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "39affa165912f7e27bb4389d866bed9edab3fa0e9b9ded3422d6e9d175f159c3",
    "sha256" + debug_suffix: "da12dc6b951e4e049217cc76c1f7a80159f9b6c932656a73118f9803d361a990",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e0e6e388dfe08d62bcc9df9289307bd85c7c712db81b9ccb1ba785cbd010f799",
    "sha256" + debug_suffix: "61b973f9d09d45e3854c1d7b4cce8761db9d2b27b3ca1600173497861e2e6b47",
  ],
  "kernels_optimized": [
    "sha256": "fde0eca72dd437c642b69dc0334731f040cda7f63375e19593889fefbf22e2b4",
    "sha256" + debug_suffix: "2f2cbd60d7d18ab74e8a93e72c5b330dfee658b61d4574328e79ae51cb878023",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f84c9185c45a8e414325420b7947270f539b6f052cadca291b1cd3475edeee27",
    "sha256" + debug_suffix: "917efbd0d62916def7c63f173185d094d37c8f1de137f3d0df99cb5b7ae6f2f6",
  ],
  "kernels_torchao": [
    "sha256": "12b3aef148b4914393751db41741dfcd31fecf1cbf6383b0dd519dae87e33cf3",
    "sha256" + debug_suffix: "2e50edf8b58c1f6590a6e667e745cec74e8dc160d2156bd3a0b267cc260f9bf7",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "3a70ba4bf8d44723d7e0fd2769eb130819b4e709ed10965f952e07c9da2e2076",
    "sha256" + debug_suffix: "e7ee3e5e918aeba11b77b0ebdcdc2773ff38b83d21442fe9fb6162a95e3eb628",
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
