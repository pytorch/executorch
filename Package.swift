// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250409"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "366d0b3d5fe66f25450e958ec35ea1b162a74f80bbe29e9288608b591422bdf5",
    "sha256" + debug: "3c5a8698094f4a50f617f3ccdfa70a23e538dfc5536c55e5a81fe06c7fd04b19",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6aafcfa62c4820e8575a18bbeba5305f12513b9f9aef58ccb300b54ef82f7f13",
    "sha256" + debug: "b7cdaf94874c972e8a5be3e7f83af759aec97d12dc9de0d331b2acfcce0ef1fe",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5fef1451c2e551e7ac916630cbfc406465e869be71153e2edbf216ae1972c872",
    "sha256" + debug: "299a0f285a40464bc3abd53bc7e32c81a7bed33d300becf2e3c89eec4df7b1e3",
  ],
  "executorch": [
    "sha256": "5cf760fb887a741aaf52949807bf19d90115729105e75c50ed4914fec6677a1e",
    "sha256" + debug: "c6079ec8072ff7ff0b5bfb01e1338fb5a408c324c486ef206b008dc9295d6701",
  ],
  "kernels_custom": [
    "sha256": "9d77415265ee3493ed2a0676b6056b5aa0c7c13c2b6c2ee8c0b9063f241e1d77",
    "sha256" + debug: "4c566ed4ae22646a309743f80b6fa12118fcafa1f286d16aded2a938791ca8e9",
  ],
  "kernels_optimized": [
    "sha256": "3919cb4d93ccdd4e8f745e288cddca3361fd7cd7bfab142462a6b330f854747a",
    "sha256" + debug: "99364b754ae2ceb20c0cdb06e8b5d14dd9c2a283705faa17d1685a9dbd963aa6",
  ],
  "kernels_portable": [
    "sha256": "6eec63301f4cbd979bd03de735791be1e61ab9d8298a624c5acca4d101df529f",
    "sha256" + debug: "51600a1d4db6bf63515f081d9943cf9fd9c36a91b071a31e96a3b5c9b82ac8f8",
  ],
  "kernels_quantized": [
    "sha256": "a40853bdbfd8d98a4b799634620fad8b2718d33abc5d04726d01f9fda5c9060d",
    "sha256" + debug: "f0acd347539ee0929a4f93724a0258c0bdccf373a171d96a00f8d006106c8e32",
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
