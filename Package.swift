// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260612"
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
    "sha256": "17440d62b479655baa4fe5c6d8d45c023ba98a9a22f3fac16aad8f1164d970b3",
    "sha256" + debug_suffix: "dae10420930eadb27368f88da55bb753a9a4a19c5b672f01b16fbf9d8fece5e5",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7354b430bf7c82d4825df0c0dfc103ea5ee273b033aca3dd66a4c3031699b2cd",
    "sha256" + debug_suffix: "dab3569a9b69209136b6973f03ed4b694cb23dc120078c2a07d5f1d6c251318e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "550896484c66521f7600a658cd440e43288958136d59b762fac3c78bdb978514",
    "sha256" + debug_suffix: "ef6b92dc77ffe555383c6bd06e0258c163f217f802283eb841d2a95c81401449",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "35609699d41e01f1b4a4a4326e67923e53bb382059826ba83a7f3cac7aa0bba8",
    "sha256" + debug_suffix: "7d325bc9e775fd4acf987ef61688ba5637452766ab45b9e957fb027036013b43",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "219a1c3d259e2140091d69187a4b105683705da96f1cf1bbd2b4370efdf0d1aa",
    "sha256" + debug_suffix: "89d78253eb7df79c6222d79016302202169b88aa87016cddcebd2a03f6504fec",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "aee2455ad4475435b2738f933d8ae75f91f8564eaf464012d0744832cf22e282",
    "sha256" + debug_suffix: "0634ac333d03eefab71fa5388f142efa4e11695d0a0e67921a38c945c5f02d7d",
  ],
  "kernels_optimized": [
    "sha256": "f7eadc848480b4ae1a253b7f855628eb771e845d33d4c207b788e4d57facf1ac",
    "sha256" + debug_suffix: "b03ead5fb25257c0d05f0313b51f5e24e0e6d1831077d5668bee6393818674e4",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "699ade4bfb20181029d985aff7095a55b41e095c59171848a4a7b92f70283dba",
    "sha256" + debug_suffix: "35c138cf900241e4e937a60d34a935a13e185f3cbb496088b5e063511b6e36c6",
  ],
  "kernels_torchao": [
    "sha256": "e9f88c9b88ce8bed1b138cb367c91f654013dc847dee161fa9a546f09ede4bdf",
    "sha256" + debug_suffix: "832af9d16c8f345f22f886b60a3c43b014d3517231006de828a0a848f69792e3",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "f8a742e3569fbebae9ed2ca38078fdf06fbb95a5ef92928b5ae5bc8119d9dcdf",
    "sha256" + debug_suffix: "00420cfbd2d4ce39132e6fab1a9c18504d3a0976e1938931f9061dd31a44ab4b",
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
