// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250403"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "4e1b467414125eda642cef590f77e24e11d19dbcba3722f6d65d1065996ba2d3",
    "sha256" + debug: "7425c084f884dddd8c5d8ff8ecaaa4a6e258b95290733fcaf0e270345b13b822",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5b3a54471b853b3b18446c819b9c17b4bbae530a139c98a0ed0445b998ec79e3",
    "sha256" + debug: "2fa24494663a135924a611030fc09b4b1fbc532f8c8f20bdc452e6a0dc191dc3",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3ebaf6597b7a771ea8b9f73c2d898dda3c39db0c373385250d7e8b2967ac7255",
    "sha256" + debug: "5ccc28e2432bb09ee3cecf7f07e8ae722ab47bb880670b91d05bb11f7574068b",
  ],
  "executorch": [
    "sha256": "ad8bee2039e491bbd46e7771289f1ad3e24134389410453195654cff20015911",
    "sha256" + debug: "1c13dcd6965312f927c3e6691144a2875e90125e5592fe72f8677f66ebc4e5e2",
  ],
  "kernels_custom": [
    "sha256": "c9f19e36cbe0d425e14d97ec1ca71058fabdde7e8df7c2d7264f840de0c7b293",
    "sha256" + debug: "b2e2f8fec72a2aacc0a425a4dcb9fc25db67a3d2e1767fd7e6e2f4c96c42544e",
  ],
  "kernels_optimized": [
    "sha256": "0e706792ec956be13d3bae3ad7a8b1423e480caf6bc2bdf9c45edec21ef9b262",
    "sha256" + debug: "c77a40f864d57f27d0a6b58cf5e3699a8a20c0deab5c8b6988f24ac6c1328dbf",
  ],
  "kernels_portable": [
    "sha256": "ad86ff554d758272af020535a6e59d4e185a3d2279f909dbcb3cf132e2f42be1",
    "sha256" + debug: "9bbeafab98a0131fd4675bc26d219c88f289f03b1e3cb920335774807c6ebe0f",
  ],
  "kernels_quantized": [
    "sha256": "96a3b3f7db605b0259ef99888ed23fe24474f6c42331b19d0eec0baa9aa5b048",
    "sha256" + debug: "5bdbafd26d58cdf3c84d5c70095ec116bd1d074f3d426f8c981d90c329a8e738",
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
