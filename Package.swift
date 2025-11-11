// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251111"
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
    "sha256": "467b72f58e539859c8c4491a635f477c325090206afe1e2ff89545b0b0ea62dc",
    "sha256" + debug_suffix: "fa32ec8b9778c11fa8cc754d6e7e74d5d7056ea1d0a2f82b740d30eaab81bbd7",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7b9428d6900e8539f28b92ae5c69d6c0cf9b5764bc10187aa6ad9bbcec48751c",
    "sha256" + debug_suffix: "c27a981814361034a83c7862b29d5a1996be1796f44bbc708527bfe3930f6843",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "cbc6eb1e029befe37d617c38a3a8a04bd8c6884e03e3bfe56a296e02d667a2e2",
    "sha256" + debug_suffix: "352b03d46ed109db67fcb8805c80a180a4d40be4b7abfbca3e0ff1f144af6c01",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "bec67db80269b737e2ebca8d13550b308f91be14a8acde7ea8aa24d8b999e358",
    "sha256" + debug_suffix: "9bb9183941bc13f24fb71203b8c39ae1c9406cef8aef65bcfebe627a2166a7f1",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "ec83b2e8cba270273c9449a12818e7ef4efbe2cee53bc349a61f949ab83dcd3f",
    "sha256" + debug_suffix: "add050e6fd1f512ec67f043e2fb1176424a759e80153d40be6bee88f3adcc907",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "26344de96c29fa5420539b7e2dba63b5509c3c167b6943f1c25b5b62fda62712",
    "sha256" + debug_suffix: "c7aa8061b6fb62aaa6d7aabf9e98c1d597fa4be8d33f076084fe1c5c7a8d6eae",
  ],
  "kernels_optimized": [
    "sha256": "2799f84b27fef1cf46fed2b5729b2a84552f19af8e8428992e8d0e02b5307344",
    "sha256" + debug_suffix: "b7596d8af5c632e0663f9c7eda083418c7a931d6de6d3bf30fc5eabacd7e1032",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ebc1c7645b34787d7a553b7a9b53e867a8aa24269c1a5bc8c169225ed24c0813",
    "sha256" + debug_suffix: "010f6827c37a2477d9528ab636ae47c3026e23e1b0fa73e07c2a58c7f50408c6",
  ],
  "kernels_torchao": [
    "sha256": "1f17fb12d27a625a769efe6586b733c28a3aaff0cd2221832a6dfee4fead0ffb",
    "sha256" + debug_suffix: "2421c591b1f692f56090b28f79fa8330ead83df121e18eaa750279584ac64520",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "3aa41830753542ad689966e42249f93f733bbea69461eb2388ce3c92856c8c9f",
    "sha256" + debug_suffix: "5abeb458e863d0349217b86e9bd7f82c7c8bd9d749c5b64d21eba021ecb09f3b",
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
