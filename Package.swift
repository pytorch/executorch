// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250724"
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
    "sha256": "e3b399eef95e6069a7fb6f1eaa5910b84fd6076a364861e2818a70d050941d20",
    "sha256" + debug_suffix: "9ba3ddc442991535c5694e43e0dacfd8537bdaae2145f6237a5c769ad092fc60",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6cf405d7ad15068a76a1e87d254db26d8432f77add527a27d75102f3bf19e0c1",
    "sha256" + debug_suffix: "e11d33dea0a9898950be58b4f7bb81a50a7873ca35010a1d4d94b76ede19654a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5340e332d866c72727f2ceb84f80953db5e9d52dc5f26c0a409da82021fc11bc",
    "sha256" + debug_suffix: "7cb61427f2b6506e286c35509ef4e2f5d5f8eec2bba30d3bb79bff2eabd2a4e0",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "2d5487ce8cf5e963b45b2f2b1a495afdae6a38c5a1651ac8f4a5eb1a8bf6df2c",
    "sha256" + debug_suffix: "7c4932c0f3a50b86f2f973dfd0e6650b7364ab988afbcf0b814dca644e45466f",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "03808c1bba9f708e09063d3a47ecff51925127e63dae47ade87ffab3616769ce",
    "sha256" + debug_suffix: "27049620ee2c5795607202c6c7b299fcecb95f5806404b6fe11c6bcc797343f4",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "29c2f746d796417d4382f6615166ae38993ac39be6f564f739b48023d6cf2ff8",
    "sha256" + debug_suffix: "82c0a901b2d11e46a13cb76bdf404dd67f90910a16e5ca889c208df87a3f7267",
  ],
  "kernels_optimized": [
    "sha256": "7f2b82fe2efa237f966a6bbd94fe5dbb76eff6fa7d132a645b39e66502e4c5e7",
    "sha256" + debug_suffix: "daad2415ec57fc5e5cd3423e859bea042ee236d806993ddcf28c7ea54a433665",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "59d5974f419152b92cee2fb21a4d4ccd5189468418106221afea3a47a9e30221",
    "sha256" + debug_suffix: "bac1439e7f188a4d60e8abf2f7417fd0ea25c144492dba5a64be866cbba2d534",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "287b1813660da3c84ea74b86f1c028e8e3b79d58f88736fd3ffb4d743cdb6623",
    "sha256" + debug_suffix: "68e544693eb84b2d854a668fad76b9fb6bf451d46d4a524f073c6df59fd3cc21",
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
