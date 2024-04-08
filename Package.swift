// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.1.0"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "coreml_backend": [
    "sha256": "0e5973bbc547e3a39f988f9a7a68b47bda0a6a17b04516fff6957fd527f8cd48",
    "sha256" + debug: "c63773f0098625f884fecb11b4a5f6318b97d566329fef8b013444829cd7c421",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "custom_backend": [
    "sha256": "c8405e21324262cd6590046096ddeb3ac33a598f88afc817a2f2fdee821da150",
    "sha256" + debug: "a08a6aa15ddce61a76cd1bf2206d017cc4ac7dcb9ca312ad7750a36814448eaa",
  ],
  "executorch": [
    "sha256": "57269f9b81d56a3d96ece2012e2ece3af24174846abd98de9a3bee07f3b9583d",
    "sha256" + debug: "66975caf3d9c1238d29945288f23ddb6e07e16386d4dedf429c0f2d81cfbe0cc",
  ],
  "mps_backend": [
    "sha256": "bb7531172252b6535429fbde429de208665f933d0f509982872eada86839e734",
    "sha256" + debug: "6d41437e40cb794b4b7a0d971931773de263370463b38a014f38e99bd1c5d52b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "optimized_backend": [
    "sha256": "bdab593fb49c9000291dbf691ad578d771883745ed2851f00492e828d089d1ea",
    "sha256" + debug: "8316ad259d6aafecf2e9abc91a04fc1fa3e0398597e043119b4c29c21e9f2029",
  ],
  "portable_backend": [
    "sha256": "38ebdad7d5cd24ca44cd950d561dcf9a9b883dff626c167bc6f5f28f041b8406",
    "sha256" + debug: "9e68b3e92e5c920875845f59821ee984b87486d05c1bf8a461b011530e02dd55",
  ],
  "quantized_backend": [
    "sha256": "245a3acbf06c6afe9cfb6b03eddfa015390e582ffdfb76efd23b7c810f080f10",
    "sha256" + debug: "134d759fe708a4ffbf7efbd25c6020186e1a13abc0dac0a897e2fe13aac3e76a",
  ],
  "xnnpack_backend": [
    "sha256": "a1c9cf8347c17f3e50e45d7f37f64ee040f0a1b0a40fa4748d90b45c4150e3b2",
    "sha256" + debug: "e92a15c2982630951e5ae5e927d548049db25d89e8b639e8901c5f4650f3a7d0",
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
