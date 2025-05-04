// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250504"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "ea35070acf26782b84e9ef377b3f229c52cc2e48d38794119777e335dbf99f7d",
    "sha256" + debug: "d749e7bf403c2dd268b802fad3edd81ca022696aa0bc57240f661c98763b5bbc",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "f351e335ad5312a96ab18bc592e20483f7856817a83072a68ce0f985fff32f50",
    "sha256" + debug: "72a43b9cc08ee487307421edc4be013bfed3fa9ae23de72a6b4b08d9e7c4fcce",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a601d45c82ed7098eaff6795081a7fe499eca4f553316bd6e964c9b8d46db63b",
    "sha256" + debug: "b03bce20860ff431fd71b6d0eca2ca7f374feacce2bbcb9e870c9ded6b438c87",
  ],
  "executorch": [
    "sha256": "efd8f027c46d266aa20a53f4a7928a1ec5ea59dad3ddbe1eaa2cf83260bce7f0",
    "sha256" + debug: "e102007244faba61399d4e141af9a6bee711a54d3ef19aa43beeb058dda06fd5",
  ],
  "kernels_custom": [
    "sha256": "5970ecbad3853524a412bfccb421358a1f7a7b6228273f31779e432c50c62793",
    "sha256" + debug: "1d3f95c8e84c6cbf0e199bb10ce264760037b4adec1ec8274f614c862404e070",
  ],
  "kernels_optimized": [
    "sha256": "db6ceb469ba2e075ad08d1d030cab26cfb8300041463e5542789d8cc74f7d012",
    "sha256" + debug: "26aa2c7e9118bf7787e852e73bf5135c37c03e7da7dcbf49c4520f8ed302babd",
  ],
  "kernels_portable": [
    "sha256": "c03afafd1c1bb4452926c23ed4169e9a7f5bda148454463d1792c29919812dbe",
    "sha256" + debug: "6477ce9e67655095c622ccd5631aab20035a2219440b1b42c0e26ad625c9ca6e",
  ],
  "kernels_quantized": [
    "sha256": "6616bb72bbad185c945eedf5415fb15cc07715d4f8142cf08d25112a2e2a4f26",
    "sha256" + debug: "b86021051a2225d9bdd5347b7284ec81ad3423f72a21178c4d8257dda77588e4",
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
        linkerSettings: [
          .linkedLibrary("c++")
        ] +
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
