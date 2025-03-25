// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250325"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "80b65ef36c06085ba384354935d0cbeaf441d3986b9495343aeeba8e847d4d8d",
    "sha256" + debug: "0ea203c6470c709859a706d3593a9dec993e39d1b9d4548dff5837429897ebd9",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "185d6d9700be3c69281462caf633a6de094e155fcd9da94eafe5aceb75576561",
    "sha256" + debug: "e2e3342bdfa061efa5a5bf066c62e942e09f1425f13630864bfa6bc60bdca31d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "72ea8112d05050408212b1b567066e0b670ae4737a083181fbbc58cd39f127fe",
    "sha256" + debug: "8989f7b506d6a36a629a7001a0b0f9debc3f94972c4f3372acd98e096a4443df",
  ],
  "executorch": [
    "sha256": "67bb24d9a56152086ebd3cb125a10007dc0fc3b71434684f9d69f000ed973d93",
    "sha256" + debug: "92c97f2f92eff845a80608c52cfda242dff06605401cc71308207b0508a808dd",
  ],
  "kernels_custom": [
    "sha256": "1ba7c6917ff8095fb23716d32875cfcd61edbaf76a5079a6c95eae49b03c40cd",
    "sha256" + debug: "bcbfa34bc6a456a33a5e528d0878836b55353a8bb7971e452f067c2d14c5df0a",
  ],
  "kernels_optimized": [
    "sha256": "73c4f06b87bcc179f0da82f6701da5b8ab6974f967a343dc5baaff57468b34c0",
    "sha256" + debug: "42e6bdf618dd3688d79ae2678dd35ab75dad2ce8a494dc50064b33f042fa5ce9",
  ],
  "kernels_portable": [
    "sha256": "c2322f7901bc99dd583375944cbcc9fca9786d8b6dd7a690bf238cc8e70a8511",
    "sha256" + debug: "50aa395849dab7e192d79436ddc2af5fb2cbb8448d54b099f681230567d2746c",
  ],
  "kernels_quantized": [
    "sha256": "d272069b36454dd3eae946b8aecf2b23c21d47ac5aed000c073bd6b085540e49",
    "sha256" + debug: "5d034a683c2eb5e0cabefe45e56a513c205a75ba00e8bf143697fc20d214e5b1",
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
