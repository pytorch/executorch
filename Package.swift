// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0"
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
    "sha256": "91fc7726cd8adf00ef6a4891e6509b0e032730844539f7c797d949032355a25d",
    "sha256" + debug_suffix: "2ecc565ebbb5a7158ffc581b4cf0f2a3ca9c255204b4c6cd6d4a3246b8efb8b9",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "13b138adeb4f885b9932554a3da6b6c9c30431772550f9e0754c37a340e0dc01",
    "sha256" + debug_suffix: "e77cfea03591b994475a13c075d0552a40e980397e077ed0fe954a9f3c1f288d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "360d6b352060cf864b1369cfb3a5ad58314598473429aadd10bfd52b43521d69",
    "sha256" + debug_suffix: "4331c40c01a754c9d10bafeeff3e6d9e7395a2dcaec908a0349bf42ec8aca5b6",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "7e9149bd565a0262dc483bea1c69fc36f87b93c8bed2a4edd51200f9ea133385",
    "sha256" + debug_suffix: "665a910931faea795496953055057d82ed4e834f464e3ceac81ef79212600180",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "__SHA256_executorch_llm__",
    "sha256" + debug_suffix: "__SHA256_executorch_llm_debug__",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "__SHA256_kernels_llm__",
    "sha256" + debug_suffix: "__SHA256_kernels_llm_debug__",
  ],
  "kernels_optimized": [
    "sha256": "f95980ba5c7ff7b3d1cd35da82cbbe63cfc23f86742029c66cf1fae647d351db",
    "sha256" + debug_suffix: "c04f5ac7ecd4fa6cdf39d2f3424cc3b0c89e8606cf9be5fd1e0e7553ef14fee1",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "dd012a9c2183bf8a0e6efbe115b1537d91fcceb67bb781f2a1e314e9297a2501",
    "sha256" + debug_suffix: "fde3ab7b6ab061783020a716593007cc95b6273c2c75917a4a1b183c37a6e517",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "1ceda8a968644b7871da8bc94ff17d6d5c9d4bb72295536fb568ba634120ea24",
    "sha256" + debug_suffix: "585c44f9f86eb9faaa1837dcfec892182b3033ae7c8106d9eb9330b3e6d8c21a",
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
