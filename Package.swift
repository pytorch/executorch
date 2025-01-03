// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250103"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "d339238b6ab5d4b343054b51b4ec86784a531cacdf1f36f66b25325c44f2265d",
    "sha256" + debug: "9637db2a7ddf445aebb5f606818f4979078898ea147170ea710a5cade3ccda11",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "03919985c63e534761bfbaa83e4959829c42e433b68566a1f79e7da09c5662e5",
    "sha256" + debug: "112eabf2bd6e3fcea7b90175098e5bb53c53959a47d9cfa0cc060b8fc662fb9f",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e6f1afc0d9c5f9b1a68ef8c6927f1d695c895412c3e411664bfd181010423833",
    "sha256" + debug: "6b2b79127609afdef8da34731bb5aa78062346963c954449e5efb52b188eb55d",
  ],
  "executorch": [
    "sha256": "f363a9448b270800180362e154571cf9ecaa44a2562f35cc45aa5e2fdb2eadb5",
    "sha256" + debug: "424e0204dd78af6b172a3d04a26d49b8db3698d456d5ed08fdb5975fddfb8cbb",
  ],
  "kernels_custom": [
    "sha256": "cbd880d12c423cde0ef4e5d9d6a8a3caae92b9652b2a18ce95fe6c92af348459",
    "sha256" + debug: "7d20efd81d0aa35f7a4bbb16d460d32d75528a516b6b3cd45ff5804ca3a3f730",
  ],
  "kernels_optimized": [
    "sha256": "29de1c906e8d540ea09857055f1a369ca35d0330d723f67a59c271fcbf88c5eb",
    "sha256" + debug: "60d0e42c5a8ef113572b94359019d9133cfc10600aa594cff548b971de812688",
  ],
  "kernels_portable": [
    "sha256": "48439941aa3726df0ac37c9ed9b07d0adf6af7bb8a22a9c39df8a5b10104ccf2",
    "sha256" + debug: "f8d67dbcd6a7e61369424ec48b766a4b023695e64e1b217d02eb187bf35cc2df",
  ],
  "kernels_quantized": [
    "sha256": "951849c9dd2ed37ed22d9bfa8853c2bb020e25e051ff1962baa637fffd587e85",
    "sha256" + debug: "790c1c6b9a6422e0d369c0dd7d5ca8ebea35d66a99b456ad08fe47b1c08f47fb",
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
