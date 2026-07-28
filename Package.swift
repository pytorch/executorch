// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260728"
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
    "sha256": "4dffacd6857b535ac02ef238e21c8f5448dedfc65e9f31cc2d98bef76c526f2b",
    "sha256" + debug_suffix: "2de8440dadd97fcc5e5e291ee6b5687fa886912439c5a3d93107c9d066070e96",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "792db9e78c95f4dbadc1f8ec4cd8c897bcde34462e9d7ca17aa806e7e95ebd91",
    "sha256" + debug_suffix: "565e96d4dbaa6924d0a204654d9c84df29b83ee65abd5e312aa0659e8e76470b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5ff3880490f12ba6f8d48ac6f09305ec422b2c907fe98f78a580cf0e0471d075",
    "sha256" + debug_suffix: "acc6dac3a22b6180e7e611e77449295abb3f34e180ade6f5b58a5b17899f38ff",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "6c391b6de6de8b5c2beebefdb22662719a3f94673c9ac44e3044013f2c48ffa1",
    "sha256" + debug_suffix: "15fe8fec55aeaf1660e1412e416324598aff91400a7b940d50217770237763ce",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "1baf5e8365fbbe6a817cc774825ace1af3664f73f35c12415d2a241430cdc349",
    "sha256" + debug_suffix: "60b7d8734360cdb689ada2644f57c6e0ea88846e7b2d7452a87cb71301eecc91",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "22f8a9fb38a55fe62d83f395a41bc59c0f90cef1f4fc1b575d7637fa279a113d",
    "sha256" + debug_suffix: "52ec72b0e6bf145fe70f245f6239c41aa3ce8755a2be8f0d5b600ac84fd96a57",
  ],
  "kernels_optimized": [
    "sha256": "428caeaa18daad9ca7f14589fffeabf55360175401276a5fd6bfd521e07725be",
    "sha256" + debug_suffix: "3a056c8026dbf9aed95fdecdeeec3cd275581b8cf94730ac3cc135ac0d53e86b",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "743ac4e5224b7be7d55cc423250a23fba6497e00a922de0681930958684f816e",
    "sha256" + debug_suffix: "de9a0f33995aefe015c1e42a21d5b4b0b7064311f9ab541c633293893116a12d",
  ],
  "kernels_torchao": [
    "sha256": "1e14c0892c1219c41eb13a71a44154c87fbb92fb2b134b39f0036a382c463454",
    "sha256" + debug_suffix: "3f53e2e10125a59b3715fbff8169b609efa11cfaa65596773cb584d4e1b8c8e8",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "24607673d5920f216ea850cb496265f4dd473f447f06174cb87f1033ff7fda16",
    "sha256" + debug_suffix: "b523d78411755420562f6b985a4c5227da5e03126a7d7496bb6665d68e6eebbe",
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
