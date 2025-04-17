// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250417"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "c5dc0363380f066bb82d39cbdd189e1f8c666cd1b406b247cb8372c67b5c8783",
    "sha256" + debug: "84a55954675d4a0c113a6b1fa2f608dead1def1e5b424f70f66266aac92e7119",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d88fe63b1552fb34a274f825659d760e2ef804defb1c3af94011b9da4ff06b83",
    "sha256" + debug: "4167d0bf4c78173a363e26597424fc715703dde4790532c699bdf83b7a95da29",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d914f0d65dd0c32300c4eaf4b303aacac0865d7fd0d095fd6caa638685f8aee3",
    "sha256" + debug: "b0ec00e35eaca8a5cfd89eb79286b5306da645c46f7fa3e38e375a1206f15c21",
  ],
  "executorch": [
    "sha256": "afc925b37aedfb5662801f157be40b6dc0ed37bcebdbdea137db838d3a4f7701",
    "sha256" + debug: "8f03c80c289aeaea0eb4264f8b6d50580ed1be7b2511ea064c6b6a1406851ae1",
  ],
  "kernels_custom": [
    "sha256": "ba67095a0a957cae7836244998a038f004d5a6fcfd24c31060cdaeb7d2acb9b7",
    "sha256" + debug: "dd4e7ac69f7d2343bfd5bae180ed48b02a99466a9f17a51e59a2593421971001",
  ],
  "kernels_optimized": [
    "sha256": "1b29f38164c9cda471ea92a670c930b501c44e59bdfaad7ea3a24e303829aecf",
    "sha256" + debug: "258fe1ae1d4937de0b2eb6bc8a4283ef356e37ba81422861841bdb30acf5ad3b",
  ],
  "kernels_portable": [
    "sha256": "6a4592b462c984af92645ecca2f34e42621b771cec54ad216d46b0ae93eaed92",
    "sha256" + debug: "60208f6ee8e9d1efa1e04a02ffa09bdfc736060976ac494f2133ad36a9b1bd4b",
  ],
  "kernels_quantized": [
    "sha256": "bbdb9242be5c80ccce7de2c9555a4edc201312bb41fb15da39049eba99818e6c",
    "sha256" + debug: "b4dfa74b257d1f2ed5e07683107058e7de58c72cae2df52088c9728aec6d9ba2",
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
