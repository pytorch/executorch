// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "latest"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "b89dca2dde55575125a577f1d546967a83c987aab26168d909526a0be5384c69",
    "sha256" + debug: "f599a096751ee3f3ef47c36b7dc8f4f7339fdd16fe6e57ec257cb5b4450360d8",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "942c32a845f761aefd04761d4e305322e4dc14a965985bbb7062015e570c917c",
    "sha256" + debug: "b45fe7a903709ad6343f6222a6c028f4d57df6bc76a5bf9238f21b58ca7a938b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "22e524b07450ed253f05df971ea2af4989e316b05a6d3aa82739b2fe48f34ee4",
    "sha256" + debug: "6fccd70fe69398ae877b6da109b31b7a2756793971e99b4db8ec1f6d4b216bdc",
  ],
  "executorch": [
    "sha256": "8cf423b2682efba22107dc2f632191aa564a50e0b6e2a9dd14a4c3c15f40115d",
    "sha256" + debug: "0032c7c50896411834d3bc720775dd0eacd0ba621cc773915058112f37673750",
  ],
  "kernels_custom": [
    "sha256": "af70560f4d391f34b56e87149a4115e5cbf438dbfd9384bea8eafa4647b6c1a7",
    "sha256" + debug: "133628cdffb7f4b00b841263aaba3029d3d5191e1ffa2c9ab66ca42c238f3708",
  ],
  "kernels_optimized": [
    "sha256": "c417dd7941b12019f5b15068d89eab0d85d0ba8aaf78585e6d79ba773090e6ae",
    "sha256" + debug: "662cff821590b987f5a8ff7affcf2f121af32519d134dbfbe963cd697c4ab5b2",
  ],
  "kernels_portable": [
    "sha256": "7a6a436b94388981d225afaf5381606ff49f78ba38d3d9a0179b68332c2c68ca",
    "sha256" + debug: "7fb31ff3d50c8df05c895c5579383513190b46b061719f831316d986f2520d67",
  ],
  "kernels_quantized": [
    "sha256": "9f8a2690f4dee47e36404c7b5b537fe710c53c21213d8dd2863599c9646a4a05",
    "sha256" + debug: "ef301c5e327e5669bed4c5553008cfb14144798edeb15742a87712c767abab49",
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
    .iOS(.v15),
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
        path: ".swift/\(key)",
        linkerSettings:
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
