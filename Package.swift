// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260730"
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
    "sha256": "9e7656c6b43872a9f6b7a462bf8b0ce8f1b044dc7cad4a8eacff41a018849cb5",
    "sha256" + debug_suffix: "8d8bc9a69a97c6876071bf01c47e753670d8ac541f3d37b6b37fdc049ef18d95",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2f08bb47ba727b158c59eb87f3d22ff207277c0a5c1093166f07af3bcbfb1c87",
    "sha256" + debug_suffix: "488f3d2f0e78e7d5c55c2b3a374b8ab405853925d8754a2f651016ce6c7810ac",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "637b498d1b59f0170da9b348b04ec3710ac2699621848f649b84c3afd878a67d",
    "sha256" + debug_suffix: "40e3f8085f2f53a5299b53ff3d37eeea90d2be7385a6c52ed5acb5bcb6d622ee",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "400cbaf94c876b6d1cf678eb7368beaec7b56bc3239dc0199aad60877d1bccf8",
    "sha256" + debug_suffix: "ee58971868db264e7329b6546f896214b019a4b48911bddc885230b55e52c9ec",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "8c3d83c2079a96804ad0a68f9848c8833b8bfaa69c6331f99b0fb7e582b68d6a",
    "sha256" + debug_suffix: "0d0b26257745c70dc6cd9a5faaed349d4c21285196fd6c003248a720a239b962",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "ae67d080b823ebf8ace73186dd7a416ff8a45ac9a884616de98d149cc5afe8bc",
    "sha256" + debug_suffix: "d4ce912e176164e465c6a97659b03214bab88a22189f1bbb9e85144ebeaa451b",
  ],
  "kernels_optimized": [
    "sha256": "428c3138840cae267adff2e2437d3e2284b18f403090b4de4986192fd107c82b",
    "sha256" + debug_suffix: "e5a082a1b02667527ae753b55c146a6e0ffec51b6b7d3780ff90679dc3b7916e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "1456ef2dc67cd8314c34e1c70d157c9ba4972f9a2bf986029278ce6baa067f55",
    "sha256" + debug_suffix: "3f9357d55552e2fa69d53a7be8979c35d097413daa1f1278f2dbd56e10451d93",
  ],
  "kernels_torchao": [
    "sha256": "dfd5c5696dfab59e7604e5afd00159204f61e5c184eaa2699d756c43b0d36c18",
    "sha256" + debug_suffix: "a7f5ee5e608be9f6af9b5b6e905bca1b81acd9b7f1161eb15cbc001ea3d9456b",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "b71fc61fe06a750d104c0a8ee1f86c97ecb308a2af4585274ce8b24aa06617e1",
    "sha256" + debug_suffix: "cc7c26f2672ce5da3ec9bd2168abd30794697dbe7746c750cabbf3ddffa7920b",
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
