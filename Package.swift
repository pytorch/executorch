// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250910"
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
    "sha256": "90d4e30b5a1f5e3bb38af8fcc330d3f47c97a7d999afa8f3b12b8e40448ab0bb",
    "sha256" + debug_suffix: "b911679ff533754d213c2c59a8378bce7070033bd18242fb5f2e1ca6508e3b1e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "fb33468871653f56075907f4209e7a02a10420d494e51c9b24636b53d5bc70a3",
    "sha256" + debug_suffix: "d11722e7c0e408077f208f55aac5133f0880ba7146f143ccb777528f72585dcd",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "eb5315a6724e1be3ab4071f892f8c8da322226b1a1dc3026df8d68801263f9e6",
    "sha256" + debug_suffix: "5816ecae23a4c544f7c3c45e58d9310260b6608aedb961bea505ad2555c764eb",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "fd30ea2048a2b9e25013b36d97d4ec9ad23cebce804a8b12fcedb9cbe3d00ff9",
    "sha256" + debug_suffix: "9197e764ef271a30ec0076ef5cbae30b5711bc89b02c26b4988b1f7dec293812",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "03dfcdffe67534a17a022b49603181fe4b1b81805394298adeb80defadcf7f56",
    "sha256" + debug_suffix: "b422ddf28136300bc74bed79405fdc73db6721be823dfa312387394d385a03b2",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "25af66222883b28c05aa9378170f15f02e1a52dc597c9a845f688faba774107e",
    "sha256" + debug_suffix: "2265c6196c7d261bb6cfee65cdb19ccb550faf470e692549eb90b4ac6bc765d1",
  ],
  "kernels_optimized": [
    "sha256": "264b14cb8dfd129662ea1cbae87012b45ba2cbd2d9e69036d04f770767cd6290",
    "sha256" + debug_suffix: "fa0516f6fff7d2d017674ec6aa352ba53840b1ea147bb9684d551b03610e8e6b",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9673bb59b2d12a41bb2d8e314bd9fa508454f4687387d342f9832d4e264ae13d",
    "sha256" + debug_suffix: "73e0b65d6a87a2a66fe2130b01a75345d6fe6fcb616ac713e790006574a1162a",
  ],
  "kernels_torchao": [
    "sha256": "aa2b36b573e91d3fe809d5e6d5d1b4bf00d9e49d2e5740c3630e70725b6f89f9",
    "sha256" + debug_suffix: "06571bec4e757522fe6d449013dcdfd18d587b0a9d92aa4378b2aadc5d11b70e",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "d1af17fe538b837eca34c035af0f1c0493c459255959b04707fbe6cd6f1cdfd4",
    "sha256" + debug_suffix: "219fec752ac1ed147b797f858fc25c24f804f0ba21a2d10e5da23f6f51b690fe",
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
