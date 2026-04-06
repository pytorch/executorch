// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260406"
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
    "sha256": "2056648898c6f0c8c8f9750c64b28bb0581e71a610d8f7d76d3ad6b98216037b",
    "sha256" + debug_suffix: "9cc26d3b934148a535a37a1751a265541e5093a26769be8642dabbf8da2faded",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0abe8c73bd4b9707caf95b680c5417bebee4c338368a9e613ca8990692f3b8d1",
    "sha256" + debug_suffix: "8309ab6f5cb7f7297771483b3cc02badce85b4c018f90a1572989c0d3df83c62",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c2b7fe6d76144aa92fc400c8be65fa203e552dc62e98fe50291cd5b3823a778c",
    "sha256" + debug_suffix: "8d7243c6c8048bfc54b25d9662fb9fbf0762fd04916c18cb1e5dc7f7d56cfd3d",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "862ee102f8651d158efb09beb03e3d8ce528982ced56c965364269ad8a4b3273",
    "sha256" + debug_suffix: "23051030519d566421f62654967bc6c26220a3d3044472236ded3ae502e98930",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "69cf9a670a543b6ca7512ee7d3a727d387ea449391aae6a7e0ce96f36c04d5f7",
    "sha256" + debug_suffix: "1923fa14559d81e2330c1d12f3798381e9cc992fd570aee84f4233e1d7c0f6c4",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f65cf590f92682132764349b39e8157ed803659b8921bb8b6721055dbf5a173c",
    "sha256" + debug_suffix: "d9cd94f70647f09a06e98592a110ec19b443d5fefc46cf352ffb926cf4e5a0eb",
  ],
  "kernels_optimized": [
    "sha256": "06b348b644480106f60066f6d2221c3c3aade7a6539fb06f60d820cb5a910696",
    "sha256" + debug_suffix: "ef47e55970d1a19b98062fd48526d9cc250fd8388acdd9cd0ac18f47cc2e7f29",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "8c0b16e951dd11fcd3bc562d3f61800eb33fc95e235cac19d0df014e44ec358a",
    "sha256" + debug_suffix: "f252d0b417a11b89c90eed8aee5a25c0325a2401c5ba1e6a325f7da20d1ac9f8",
  ],
  "kernels_torchao": [
    "sha256": "96e6b9fd8c739e5d75ce366090c8fb086b19370d735ed733c6c5521bd6146a80",
    "sha256" + debug_suffix: "8b445be987ae429a9a324354c20feaa0d0bd956dea96fb657d0300f5a397995d",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "9c162cbac7ba12610b9868029ff287aa116870879879186e8e5ad834273fad46",
    "sha256" + debug_suffix: "fafbeaaf220039f4d259f34cff1aa5369087268a95fca327ff96b94dcbc257b9",
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
