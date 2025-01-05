// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250105"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "f17bb07e2c374cb4aed707a85065e491c6707c0216426f6da5d903e6020ead7b",
    "sha256" + debug: "28b8551b26a6ec1f5e75aaf7efe07519ccb5b39af4f8bbfb177a42d80e397295",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "f7ee084ecc510711dc5b1f5e976c02578dac69d893e04f6f445f44507a0c2554",
    "sha256" + debug: "7acf6978b11a4d719c541864919266db8d3617987672fc6bb988e66ba0cc27fb",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "9d9445dac04f8e2d97314db1ce055a3cc5bc0eb3f9bbc64407548003cbf30433",
    "sha256" + debug: "eb99baff38cb81019979431a91762362b486c7dc4f5a1479fd3bd81c23073e6c",
  ],
  "executorch": [
    "sha256": "f961a9b983a71c44c9ae31e0fd627f6f72448fa6b44e72af82dcce28f14b71c4",
    "sha256" + debug: "8ad5544193aeb12f9527055b954da1c890703b981c2ed34df6d062e8df93c3b8",
  ],
  "kernels_custom": [
    "sha256": "845d7c3fd203c8b8682a1018dc7f4f4de985e81628db1e38f79474697f1949b5",
    "sha256" + debug: "ab82412cbc9b021eccee48206f86a7f6878d48dadbb5ff318d8faf551e4dadbc",
  ],
  "kernels_optimized": [
    "sha256": "c867c301521df5cc4a86be3ff99b336d3b9ba0bfac7c8d36cd8f7585146dacae",
    "sha256" + debug: "658205ee3b7a035217ab34ef5be85baaa8fb6db57e998ce90bbaae89e655b8fa",
  ],
  "kernels_portable": [
    "sha256": "311349a3bb875d419925878737e405b408df081fa89f88add012bbd5d060ab11",
    "sha256" + debug: "bd6b1befbe9221929a9c70dd9f2baf7c3a1093bc38145a068e0c956773863234",
  ],
  "kernels_quantized": [
    "sha256": "3bb431d332bfb1a5e48fc3036c2e7f420e6e7d6b9f9215aebb839f0528bd2ed1",
    "sha256" + debug: "9f8de743d447d4bd944d7f7a59aea4fbe8ec40beabdd53f694417623105594b8",
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
