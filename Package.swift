// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251204"
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
    "sha256": "8937f2a54dc3d673fac02e6aa2d57b100caa73afafe9064f4c833d3852f966c6",
    "sha256" + debug_suffix: "9aeed955a84d99ea2e9bab24d703155482d46b590ab7fdc08dffda12bc97fa07",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7e7c9769f882e82fcf7ccc2490cdf6e8635436d66484dfbc0f3930bf4e1613e3",
    "sha256" + debug_suffix: "f22b09bc0c65f78507b44540f11639e17a47432ff11b764a6b9d7978a0f86393",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b5b7bfab92038a018ae74b3d071417b467b8cdbb2f84ac7d2c7639739746c1a8",
    "sha256" + debug_suffix: "3fa733066ee8fa4dbdedd8f2c75bbc8a63497d5c3214d431495c64da55faef26",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9edcdb19a7d2d091f937a7af5def0b6965e8d96bf5c218f562bc5637ee91e91f",
    "sha256" + debug_suffix: "b097a15ee99f1f26f56449efc98481acc1544dade4617716e34d46c8c07409e7",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a8f45531b1c0a94d820f34faf6772038e7ea601b0a6655c0e4d04bb10226f74a",
    "sha256" + debug_suffix: "8286be09443bb264c79cc7bf7598867578c6dcf4d8cfd740ebc4c8ae31cfa8c1",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e81a04aeb1d11d526ebb9ba221caf340d185662fa5574fe02ac6edc42f607c49",
    "sha256" + debug_suffix: "06d41cad55bc9bb190ef9df14769db19acbc49d42967c89fad500aafcf0191fc",
  ],
  "kernels_optimized": [
    "sha256": "cfc90e8de1fb5a7cfb4b7ee404492b1d91a1c8d5abf7bb112ebef700635fc01e",
    "sha256" + debug_suffix: "f375419e029026e7cccbc05ed10172a2bbabef9de3e864578a059a05d1f012f2",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "931df8e35cb869aaf8d7d932bf0a34ce4d25e0a34251f44de52c6aa3a95bf447",
    "sha256" + debug_suffix: "979c19a53d19e9a8e614052f36e2b1e7320e822ddcc72240f3f2095ae0a30511",
  ],
  "kernels_torchao": [
    "sha256": "2bc0d0713b0f9e4de719a8a4e40f3965268808f06cb1c4a1ac43667cad7bc086",
    "sha256" + debug_suffix: "5f37778d75906e051d2e7f59212e89b16719dbba237203764267b52105744ddd",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "dcf57ccbf88332190222b81504b1fc9458d732c2290d29621afc6e8ad5483a86",
    "sha256" + debug_suffix: "21e13a4bbff3de211e2376059d84f81fafd1f956619490ff786c7ad10ba66f27",
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
