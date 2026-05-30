// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260530"
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
    "sha256": "54276c34395365b6305c3f24cf516f3b27693f51e45b7328d4ac423eafa60938",
    "sha256" + debug_suffix: "4f937f78bca1f932253430ca781998982f75eb513e46244085f16fdcbeec3b0c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "50fc399fe5d87bcd3f28ea1eecea6324c6a9276a779c833b9cb2a1241c5bacb2",
    "sha256" + debug_suffix: "33987e92dbe2dc1620f738377e2a289f3d75f5f15fae397a1021de41ffb0bd86",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "425015d03f55be756bb2b8c5d32fd8c71d5cbb778c73a8a93ee87183b2a75f99",
    "sha256" + debug_suffix: "205e9245636e3b75b9c20f4dab3ab53ffdee8daf58e0c6fff9ea54b7dc65a210",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "260ebbc6308753d499f1d2e0b2a8639cfdc62a1040feb0938450271463231fdd",
    "sha256" + debug_suffix: "7c0a985721824df3a2da36491f68e5c18375db5e808b5ad265c220b99aebab7a",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "23de2fde13541df3fd1419f8f65af429736431fd1c5e270190f6babfc9aa2151",
    "sha256" + debug_suffix: "a13ff354e8609695c0c7df3b2a3b060218a5a21318a5205fa31f1ded79224873",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "88227c7725af0f21672bc65cce7af1e2fb4fddb8b00852f5301fd1be87f3fd4c",
    "sha256" + debug_suffix: "357caa1a834346f0957aa7aa93909d4ec9a1d5cb374a0345bbc585689042eca0",
  ],
  "kernels_optimized": [
    "sha256": "767a2fb372c5fb24305ebd5178adbbd6ef965dc98895c54f3349f5a8c4816483",
    "sha256" + debug_suffix: "f04c467c02f677f26d2a03eadda5c1a5bf1482a71f985cd1369cdc72c6a4ad5d",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "7468cec757a7641afdc23e4c1128129100bb44294d7d3715c61711a904bd1c57",
    "sha256" + debug_suffix: "5f6e709ba181fd41dca8e3dc99aa8d459a52c03682f5f017b3f72c5da7441d40",
  ],
  "kernels_torchao": [
    "sha256": "b22404e99d6b0bf2f4298e19fb6d90547dd118e336c4d5f9e4654ef5f85c3d31",
    "sha256" + debug_suffix: "d41ac6c7366f007cda6b899bd4898210a9f066c6d9778222a98ce12a52e071e9",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e4ad3f35922d94cde7aced5bee18c0c16c8315ce9d510df61d760c54ef19ca49",
    "sha256" + debug_suffix: "e55caf689686c7d6401d2d02010fda5b13e46154d23bd44e5259ed8556c22142",
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
