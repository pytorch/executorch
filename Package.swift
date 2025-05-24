// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250524"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "e7d77b78d41f12812596bb01b0181a7299635f190de0bc78b0c0bbe0195f4021",
    "sha256" + debug: "144096e1ca9c1dabfebaa6fb1869e37809b9d0b185655077c856e8ee15034ee0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "825403588a4b72d428aed8a2ae42e436efb12c7bcc029debd754b4fb92d6d1e2",
    "sha256" + debug: "fb36db6095423878b37210a1510fe3353de494da71400b5d1be3af9ff049e50b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "edda1fdb738bbf32f4cb6f9d7cdd043d5dae22494af6e34e06cda49d1fdbec38",
    "sha256" + debug: "7a104767218954b78b275ac44cc671bbb5cf17cdd263b69b31950ad6cfe97352",
  ],
  "executorch": [
    "sha256": "4265210aeaee7bf1d02873e86b73de84c9ff1988be21634f499f15f151b4e998",
    "sha256" + debug: "30a7814654631d8b3b0e279f2ac40f0b3bde8fd5f696c8b77a1bf8acc8943642",
  ],
  "kernels_custom": [
    "sha256": "bdec543109bfa8468aa4657f0688dcc496675a11e9d54788048c3da6b8ae8b6b",
    "sha256" + debug: "88b49f3ca502da5e62db416209239bba8e9f977765b3ae8cef9092a554ca2a31",
  ],
  "kernels_optimized": [
    "sha256": "eefc5396520d87aaf3fc4f7ee6bd3a96bb5a7761120ffed55442271e068af5f1",
    "sha256" + debug: "8541911a8f19b4dc749fd51cc7d2b6c3f0976abac8462b91e7074120a77f40a6",
  ],
  "kernels_portable": [
    "sha256": "fa54368a1518e7580a0f25e62c074462c62e395ed462e55d43a2233506f78bb9",
    "sha256" + debug: "02a756352b0ab15131ee8de9059df1e880da15fd6450cf5eec7d3149fa23c0d3",
  ],
  "kernels_quantized": [
    "sha256": "5bf3a8d4bb7d4451fe2ec709dbc371ce9cf2fedf8ddaa6234e0d3b62852b9285",
    "sha256" + debug: "307e446bd58991c4998c68c0cf30668eb75033712e6c5b8339ddae51212e9654",
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
