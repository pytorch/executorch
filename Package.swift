// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250503"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "a6a0ed3879fb2171c9d6ba765768547c1e8d84409b806b4c02a239add01cbdb6",
    "sha256" + debug: "1562052d9b08c6b835c3539b43e2446e64a1c3e15d94af6f52e0f00e8ecd25c5",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4794fd8a51e03ec0f52f6a10136a00dba1008ebacca7728f97dc90bf83c4e681",
    "sha256" + debug: "f8872efadc4947f21821dfdfa88c879b8e0d11555199e925cc3ba8fbf1302645",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e810512ef473405d54ef76b83c4ca1a0342845df2eda543eae80f01051991097",
    "sha256" + debug: "0945e07c4863b789418744eda8c48369182f9a901ee22c91eb092eb035fcc20a",
  ],
  "executorch": [
    "sha256": "49ac0d129fd1a4de0e86c11e07d4c174ddfb88a8fb1cd0df1a9fbc82143de045",
    "sha256" + debug: "2688ff4ac4c7af8da22496c76c55bc0e6a14bad46c734767d19a41872063b1e3",
  ],
  "kernels_custom": [
    "sha256": "f2772d2e8ec85b55315550f27429f231ee0090c386621d5ef9bec6de471446c3",
    "sha256" + debug: "6cbc1a2888bd42c262f5ef44acb090274c27a05cf9388a4970125a4bf3b2f784",
  ],
  "kernels_optimized": [
    "sha256": "7888d6b90fa404e39719018dc22d552dad9b8eb74175eae480e293ba9a71f24f",
    "sha256" + debug: "e62cb10d8a066a92c75331b5e11161297fa91c53455d1b29fbcf006a90de006b",
  ],
  "kernels_portable": [
    "sha256": "f6b320abfceda73dc42fac1af7033a1f12d8776220c8c8cd650f9af0123e1567",
    "sha256" + debug: "363b60147ede4d087d46e1b037061ec5d5fb7a003db16fb648c126971163b000",
  ],
  "kernels_quantized": [
    "sha256": "b694e082c7bc6e5b8d2dbe18a4f265e8d5ecda0bc074e40c3de668decfcfa2a2",
    "sha256" + debug: "efce06e511ba8faadc57d37304b2c677c17498c8f93c970e1207e42322e9b45c",
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
