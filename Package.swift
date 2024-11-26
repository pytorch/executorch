// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241126"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "b71ac88ea18a22983d48f431a7dbf9f497178dbad8e2a2961fbe5e7cf266f285",
    "sha256" + debug: "bcc04732fa75c6e6e97d88fc8fcc6e64ed30f15d90895172104efbc9ef4c6ef0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2be92110be2b4b257009549eb534af816b8d44ec5dfa804f92c916253392b756",
    "sha256" + debug: "6e1b7c6688efcb95f49f1dbed0dcc961e4b0c8e7c470df773cbba9d41e0ff32d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a138c83415fb837e318ddc769a18acf4215654d1efccbd896275bd7a530784a9",
    "sha256" + debug: "8c1e95fcd2e043bd565810de1310d0fa2600ebc6582d59fae82bf0b5839bd490",
  ],
  "executorch": [
    "sha256": "32251d3fd6b450f816c66b6eebfe3b0b2d7338c557526c4a549b45f61a538e5e",
    "sha256" + debug: "f41a72bb98825d19a7bcbc68b46c03974b53e114b738e1a75b9c03a1f85f294d",
  ],
  "kernels_custom": [
    "sha256": "7be8ccc9157b8ad84fae17e14a1469515fb9a9bc3eba2ee7154112445e4e73f0",
    "sha256" + debug: "30c4c2ad7a63fd837a40524ffb692efbd3c60d8cf5716bde356e403319420e73",
  ],
  "kernels_optimized": [
    "sha256": "ba9f05438f5c2dc0a2a8325c553132ca9d459f56d84ec1241f4ac820aba3ea69",
    "sha256" + debug: "4156e3dcb0a94043eca8c7be9c48e0974abb40dec02d8eaaf03c5b789c36449f",
  ],
  "kernels_portable": [
    "sha256": "c407dd563859a3ed7bfa0507f145ade3806a492a8e96d76ba542a57664707633",
    "sha256" + debug: "8da1b8f8c879df666737e25cd7b7b2e853db542540738fe0c323a467630b355f",
  ],
  "kernels_quantized": [
    "sha256": "62632b8d3c87f0cbf41bcc142e19ca199f327fee77a36772358a9d4f671b876e",
    "sha256" + debug: "498813fa70a5c249745ddb0b9b0ef327e88d89c1d627bb27a69148e85f8d5d24",
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
        path: ".swift/\(key)",
        linkerSettings:
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
