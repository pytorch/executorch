// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250123"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "0d5c470a60c708506d57d1970d050d998344ebad39479d33f39ea47b0f17ae3c",
    "sha256" + debug: "f559f7afe7557585d324673fead82ca8a5a4c27132874610aad516bace8240ec",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4c452ca8831fa5288eb7a1a2b873aad27dd2f14157a3c734f5512f8d64e627c8",
    "sha256" + debug: "da24f677958e407629c087becc045c15b09f744999bc8fcbb913172381af5d87",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0fde73a86a94e34222b5f6d8dd579865dc2d20c8aab00eaa37d74f871b5d7b40",
    "sha256" + debug: "43b19b288cb0563c4c4f07ea5c13ae0f36b01af5ba94dc259008f49be7f82dd3",
  ],
  "executorch": [
    "sha256": "e42a4bc693fa13cdea7bedd5ab9cc1758b3baf0c5fa134373229b62407d7d2de",
    "sha256" + debug: "eeed6798371de6c5357754715597ddc8432bdfb33326cec07474209513da2eea",
  ],
  "kernels_custom": [
    "sha256": "cc8801b35872158b637db8aedb3e16eba90679e5330e9365a2820c5609926773",
    "sha256" + debug: "3cb872bb93d80f18fabe170c209df8fcc79e858681a94a4d9fea4024d9e29de6",
  ],
  "kernels_optimized": [
    "sha256": "4e9bebf67a4a2219152f65340bd7b7f16e034b706d28f35affac8b9c01feb5c0",
    "sha256" + debug: "56bdb689ca8d3280662ab64066221169e357ebed73d62bc133b1da8f900397cd",
  ],
  "kernels_portable": [
    "sha256": "f44926027d677c3947f09dc9a33f12330048d984f63c9c1418204a76b8898d6a",
    "sha256" + debug: "6877ef7448b9c6285880531f64e3f692879bff68e9806d6181dcd4255f515705",
  ],
  "kernels_quantized": [
    "sha256": "ead8214fa58a68f7bbccb544ce0cb188b2daf67bab53d57931150c66d00a6f77",
    "sha256" + debug: "3017e57792a1f9c1a421179a3138ba380bb181d4522790cda73270ff499ad98c",
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
