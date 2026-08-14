// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.1"
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
    "sha256": "9ff34810bee54047b02dabde3c98099ebe41fa49ea35b408adb62ac06415a5ea",
    "sha256" + debug_suffix: "19eeebdb59db1d440e695d7edabf748c2106779472b12aa63d1c51672c217c6e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b2c0cc868b6902654cfbeca7ae3fc97d3ff1146b66e9dd718027a509f812b754",
    "sha256" + debug_suffix: "29e661d27c0e6bc067580b0b853d871a775291834315855913e1798e3d045da9",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "14ffb372e29c09ef48ce6de4d02da315c237517438c07c98fa61fb89cbc4c8a6",
    "sha256" + debug_suffix: "87d755425244b31dba571238ba412e900aa841bcc09d3402afc8e5a11aabd678",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "8417045ee1761a410098d7b54a1ac0e3c227ccbdcca79bf7dfa26c0cf55f1f3d",
    "sha256" + debug_suffix: "c1031a671521a11a489728817fe531b327e0adb1b537dc3013155be5db7a8394",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "d934d48e0d635f2615c3c13b6a91b3342ed26b15e509529782f26869c3af01a3",
    "sha256" + debug_suffix: "f94664a7d6394e64bcd6a3eb9baef978169df619d6af5f40256598f0340e9ff5",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "47849dc0fe3d9450d0b07f0ed2f9e3aa69267423ebdeb23423df4d980f0acd52",
    "sha256" + debug_suffix: "4b9b4aff2f5895615a4122957e22f0622dd77793198767a69c695259d3af372f",
  ],
  "kernels_optimized": [
    "sha256": "cdb5c6de303331dedaa495d429b1420899d85489b8116a2520a0433d64ef57e2",
    "sha256" + debug_suffix: "bb54b5cdb16819184799da4f01408ae10fc48a61562872747f20c6ae699311ba",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "e26a1b2e7eb9d3b5b6ae4f9fe2233cdc820ec8a85078cb61bd914e8bb19bff3b",
    "sha256" + debug_suffix: "d6851c00ce10fe8227cd91c27de1c7a10abc0aa996ad9e2164d00c7231350fca",
  ],
  "kernels_torchao": [
    "sha256": "9285341b86cd12c4d4fd850f6d4f243f0f35fa8c37a31ca764e44434aef6e7fa",
    "sha256" + debug_suffix: "15a438b04fadea39d3dc34fbcc6078db7c735e536d7d1dd334f5ad7d845f9d5c",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "d6c700c5c2307c8a8ed4416c0f8fab6d5123d6b7c06ba77fdb4265aa33c19625",
    "sha256" + debug_suffix: "c13f76bfce1391b03c708fdba812d95dc3195ffb1c5fdc4ace8f6d5ffaac925d",
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
