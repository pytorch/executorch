// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250331"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "8cef8789cb701b92f1d05e95f02024d7409392dd87f8ea1c6ab754ad55b32a78",
    "sha256" + debug: "459713d69ad3e3c3527e3bf4f28a02017fa293db21a861d5c6e9b0290f205338",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e0dc523a66cc0e9a49df8e010f7bfc6632d0c04d0361f2c7aa51589f8adfd7e7",
    "sha256" + debug: "4f01e558ac281165fb2089f6d7d7950434f16337ae66e630d0685093b97ef57e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "870ac6718bf024bbf177015aa16b1ac8cc2ecb50abef286ac2781b3876788d10",
    "sha256" + debug: "db10f12fb437bf60c28b071220234fab0c7e303121a15a5be2c2cae438f637fa",
  ],
  "executorch": [
    "sha256": "98165b94e7df4098fe0cb32a0380ee4c2988084a7898f36826eb035ab5777e3d",
    "sha256" + debug: "ed0e8aed79a54927c55423df8120db2b371563a6e7210cf3ba592f8b01ce1ce8",
  ],
  "kernels_custom": [
    "sha256": "1cf707352ce2efdae7cbafd2d89a1c8170c54c1144c5315738cf7b8b1495ade0",
    "sha256" + debug: "62f7ac7817b860be8daa4a4978c7cf6ffe4ea47c38e6cff2d0c4d10d078840ce",
  ],
  "kernels_optimized": [
    "sha256": "aab7474fb6c59c2a9866a62e8dde02ea2a9c116b5197179bb870c7677de49ac9",
    "sha256" + debug: "bd99d965ff9f9e5c57baf784a252377d4d7be0096bb3523b2dcc2d3213b4b0cc",
  ],
  "kernels_portable": [
    "sha256": "82b8caa3705d5307e63b9dc0d8f416a35a2f67febe0288237dad8deb15669b80",
    "sha256" + debug: "9cd369c5826d31b901b9a7cbba66a5857d88e1e3bc9a3f159f9474b41151d87e",
  ],
  "kernels_quantized": [
    "sha256": "e0c496727c0505c06a7ce222081f25a31fa048515c52c86e920ffc584cb3edea",
    "sha256" + debug: "28c96e33e6d6b5a10c257afcbdee1a706ad3443c225920904d7b7e3e67396f3c",
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
