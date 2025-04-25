// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250425"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "a9c9030ea127d18854c1d09e2de747c8c3c22574eaa418032311743ad5d1f089",
    "sha256" + debug: "bb1dd478410295fc34c4ae5119fdefb80bf041da0ebfbf923cbbfdd01dc3bc11",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1989faefb80d266953a56f2633294c1a6549bb2ddf4c7d792dc1128fc8d544c3",
    "sha256" + debug: "f8580a5a68dd72e2afaf669e59ebd82eac03ca8e9563fdbd3c0ca7a1915a8d83",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1a0e810f399c6b38b08343423d3883fc695d30fceb93f8b125ddc72b2a083bac",
    "sha256" + debug: "cfd005aea65d29bdd8c70fb57b9472fe2e491e5921bc8a974d92d622e6fb1b4e",
  ],
  "executorch": [
    "sha256": "b6510f196f471662f4d1a9a184061aea216cef9fc288bffff3ff2195e081d360",
    "sha256" + debug: "39e7885afc35660970d13173a151cd33c7229533454ad7330af7f8e194db86f0",
  ],
  "kernels_custom": [
    "sha256": "d42e498a3a9939f5d328710d451d5106c5709812f736a9440214aeed48f82e70",
    "sha256" + debug: "7639eef87b24e152bece52105900175977bd69c19953a966d117e72bab23d746",
  ],
  "kernels_optimized": [
    "sha256": "1d8184523c4d64066750469b64391aa52622c878491351658567009b1de4b451",
    "sha256" + debug: "0b441d33ff500904e75429abb7b11ccafa484ce2e9036d243b6dfdc98fb843cb",
  ],
  "kernels_portable": [
    "sha256": "cbe50a3219d5e3c6e1923de895e88f631f4ce9978e591d06aeda4b5ffdacfd70",
    "sha256" + debug: "08628d9b8970abfcdb4f786834973af1c8e8e66929a54cb226030bde4a2bdf80",
  ],
  "kernels_quantized": [
    "sha256": "e54240367ac91723e7cd2c606576e5697638792bdea76e73a383750f81e10fe9",
    "sha256" + debug: "11c454ff3ce4e8d30059e6f8f135b2b42154624607514bc5be47a65e84b49995",
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
