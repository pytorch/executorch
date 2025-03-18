// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250318"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "66a521e15320a4a4329c8efc0a6c0be4f4fbbb9f4196abeb9bff6923f392eefa",
    "sha256" + debug: "4a43f627e9f0ce9d033803e5a94a8f74920b017f2544954eaeb7d7e5033abb3e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6b57f349c30c5e712d33d1f801dbef53b14e4714fbce9f6f971330f646353d0d",
    "sha256" + debug: "04724e5f8c46323a9778590bdf1c60b75796d990a846a9ff583a50bfebdfaace",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c8d7fb7ed1f4657c32bea894c0d3eed7766873eaa733febf7ff9c60322c8f798",
    "sha256" + debug: "5e289ae58611fe19852f46274bc9c56d14eab74caf775f3a1894b528d7a1c9c8",
  ],
  "executorch": [
    "sha256": "ad3864efbd16a038c305c9ee2afc7810406a620bbda0f3f771c101e14f729d78",
    "sha256" + debug: "0f864f82d4c05d2b7d96e81b2840b792977401a98e6c35c578fd0d45ae93aae0",
  ],
  "kernels_custom": [
    "sha256": "4733376a2d4ec8d8e6ab4d1881f3204d496ae2d0f0860950be585ffbb68d3746",
    "sha256" + debug: "585073fc198faa3062884bdd2f40cebd7526942deeaffcb8a02bad3265fd4863",
  ],
  "kernels_optimized": [
    "sha256": "653e10d7f846f0c541404d5932df032e17de791b4f1c727bcead4e849a8e5d57",
    "sha256" + debug: "af764c9a83167df414358efa553736fe6024fa18cc901e73aa46347a86e6a63c",
  ],
  "kernels_portable": [
    "sha256": "1ad21d4dd3db7729142a733ab6f2343d1247fafce945555c79c2f8d3db6bdf8c",
    "sha256" + debug: "496220547acbe14ed95cebda022f82946d2d0650d59878e72013a72a72087973",
  ],
  "kernels_quantized": [
    "sha256": "92098a9d1a41793d7ac27e1c86f06927d28571924d1cf7a4a6b2746bcb53873c",
    "sha256" + debug: "79e51be0b109068927a06b33f1b0619e48ae510b19352dbb91cfa82623dac737",
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
