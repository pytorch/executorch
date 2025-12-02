// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251202"
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
    "sha256": "e4ec742258b511f7edd0c6314e9177d0b216511437cdabbc14f0366d1c5653ff",
    "sha256" + debug_suffix: "e9cd24676a18fcaa972b917c4768b73192daa60f50661dcf7c415fa4aeb965b7",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "657ba7e0b1bf91acbaf7baaa8fdab847b7a952df5cfd54e7f073bc591c07f4a9",
    "sha256" + debug_suffix: "82a5fb8a6e589fe0a76f99b54f4b2aea414cf99e598b3407e59ae39119fe27e9",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7c2a8db400bd4a392a98d4cee0863fcbfb49220b44f2fa0e942a341306bdc697",
    "sha256" + debug_suffix: "515aff8e5e16235a1e6ec426de3243a9da174948857a976dfc242c12413e93db",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "3ff9b18e87f44ac5861df30cbc8451478a98000e8b068555c20ab7870b5dec33",
    "sha256" + debug_suffix: "d09647490e6fa834d00b99ec3779e5a32c07857ef8bd7dbdab01c7a2461b6ecc",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "cdfbc0e0eb8a94b9ad7a366db6f2efb468554f13a272701a120057806a54a315",
    "sha256" + debug_suffix: "a06bd0a40e48d8cd28f4e4940f237048cb539766fdc1aeb78bac18cbe399c2f2",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "73f7d71c995216af8101b03b20afee5960be40966a07c14e4448cd89521e8647",
    "sha256" + debug_suffix: "0bb7e46397d0b693b756a5d696cdef50c22be6a4eb410a939fa31ab3fa2f2221",
  ],
  "kernels_optimized": [
    "sha256": "dc929f892c7d3585a50bdff6b3083b1b5aab0d8ed75987c810872f0349a739d8",
    "sha256" + debug_suffix: "a6b60e3ad6ddd0c232104a270f231e6fde95e39d53d016fe489e836dfa584103",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d2f82131ddbd9e3fd3377222d8509191ba04cc10c828a88582bd34c899e3b1b0",
    "sha256" + debug_suffix: "2cbab70e1f366bead6ab4ada31cca55fc142533da6e522d9193a48227d5df51a",
  ],
  "kernels_torchao": [
    "sha256": "ebb6f46482b11840cd8d84f44cba45856d0b12057f6b43976270fdda0c9b17be",
    "sha256" + debug_suffix: "9d79805b4f34adb2e2fc86ba58b113288263f868ab507af1138ed1423a0b3cde",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "877984caae951f3943b42d30987c4e36c3e980c7437002977c82fe8bcddd2bf6",
    "sha256" + debug_suffix: "6f665fc7200c5ab756b18c9b6b3321f6cf147348391e059a098b62cd5c42f5ad",
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
