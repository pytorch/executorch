// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250314"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "a2e6fd826cc4085faf89286fec55dec212dda947e6db45669fc9570389642162",
    "sha256" + debug: "aba6245a4773fe27346c14e763bdde461e0e06259b196ec9c0a306ec2dba49d9",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b6f74c3158cfa2c32cca245c6cb68d8104ab00cdee72d220020615e7c8c2ea6f",
    "sha256" + debug: "394ad20cb2037a26fc4941d377ed284a1197fc49b245571649ea09026e1abea2",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "9ed3c38364e8026d4ee9efe9fb9d21ea54173c59dfd687e83516ade3ffc35d8b",
    "sha256" + debug: "f2cf8607b79b3caea92fca4a32a5c4c02366f12b60f01b000a22ed5e2f829167",
  ],
  "executorch": [
    "sha256": "37c1d81a892624d0953c686df82f714dee028ade867254be0088984db6ac831a",
    "sha256" + debug: "e65317f0c1adc3422b5df5a33b816b9b7faf8a07e3e7146df7833dd8d504bd1e",
  ],
  "kernels_custom": [
    "sha256": "055226e9eff00c553b3b4647decb081a466b571cccad4ad90215e0eb09dea9ac",
    "sha256" + debug: "7cfdac84ca231e34ca4f6b17b6d9c8117a4ebf333dc4337159202d88d662e687",
  ],
  "kernels_optimized": [
    "sha256": "9630f9d455c33bc194aea0eb794436f9cc60deed19eccd9bb3868b584bcee337",
    "sha256" + debug: "c0d8d1ce998ffb061275fbe9c732190524457f3cbdcfb692fbf8e454b91a4764",
  ],
  "kernels_portable": [
    "sha256": "3ac6deafc4393f4bdf5c04ef2dce11b69c7065b8fd1f133378fc58d3283a6962",
    "sha256" + debug: "aa63031bc843a84de3cb9de47d73c881f78a643b6efe65c3d1f7d2aeb148e457",
  ],
  "kernels_quantized": [
    "sha256": "fb54aa4bc2c0608a8f0db803e85e8b390db0bbdf4bb7765769ef46e1ce01ec77",
    "sha256" + debug: "07a3f23fb60667ec019d0d4f6ace77df71578c9b8bdd72767ca781148108365c",
  ],
].reduce(into: [String: [String: Any]]()) {
  $0[$1.key] = $1.value
  $0[$1.key + debug] = $1.value
}
.reduce(into: [String: [String: Any]]()) {
  var newValue = $1.value
  if $1.key.hasSuffix(debug) {
    $1.value.forEach { key, value in
      if key.hasSuffix(debug) {
        newValue[String(key.dropLast(debug.count))] = value
      }
    }
  }
  $0[$1.key] = newValue.filter { key, _ in !key.hasSuffix(debug) }
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v10_15),
  ],
  products: deliverables.keys.map { key in
    .library(name: key, targets: ["\(key)_dependencies"])
  }.sorted { $0.name < $1.name },
  targets: deliverables.flatMap { key, value -> [Target] in
    [
      .binaryTarget(
        name: key,
        url: "\(url)\(key)-\(version).zip",
        checksum: value["sha256"] as? String ?? ""
      ),
      .target(
        name: "\(key)_dependencies",
        dependencies: [.target(name: key)],
        path: ".Package.swift/\(key)",
        linkerSettings:
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
