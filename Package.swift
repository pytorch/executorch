// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250301"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "311723fa5ad32804f7a54b188e00073aacf866c5a40d993a0a6ebef20098c577",
    "sha256" + debug: "64a57a1c97a68f8ceba72df31f0a9331cf841befa390daf7a0e5940c877f267a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4b62a6b735351896bd1fe8c89bf3899e0e8bc93a506793594a95460fc3e7645b",
    "sha256" + debug: "f724a439a5e6bf757ff8cf598c4706d5536a49b9886cc4aa17900056cd02a088",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "bb641553b959992a73196d1939e5276e28fa6946d0dd157e7c063993fdcb707b",
    "sha256" + debug: "1cb14ee4fb46bfd012086f4704c21d3c26a7686c32bf8613571d799a78a4de4b",
  ],
  "executorch": [
    "sha256": "1d114d2051a1947bbd712991065767d98ca90d486e8e88454dc08dfee4e54a00",
    "sha256" + debug: "35795a73e461185757a362eb9a652a355e91b7453a6491ea67987ba66fa09ac5",
  ],
  "kernels_custom": [
    "sha256": "e904f0e21f70b9af85c25e65bb771c157e43c7da8346643b820f3ce4463168cc",
    "sha256" + debug: "2707b69894f0354e9dce767d92d2c03188e51c0d128fd243cd0cac3177c5b2a8",
  ],
  "kernels_optimized": [
    "sha256": "3913f0f8655e6fbaaeacb63d0b1f3dd14d920d91cf3544a1f17606824c0cb4f1",
    "sha256" + debug: "04d12cab48014a2718f6d78d81e3a20b040245e14f8156cd96926059b85cf216",
  ],
  "kernels_portable": [
    "sha256": "f5b88851dac093b64a0b243e0ff2ad3f2d90878ae0754c5507b0374708e293e8",
    "sha256" + debug: "e4f53179769d258b43441a489c8e244569b36d1887cf5f41c46cc4283afb0c59",
  ],
  "kernels_quantized": [
    "sha256": "9e9ce90ee51fa471fee9df8f4bf7b3875f2b5ec0068ddacc60eb2714e826849a",
    "sha256" + debug: "cde791dc7176a826dfaae4c9547154ef4b165f6b473ffde1415ac16a80bf8f31",
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
