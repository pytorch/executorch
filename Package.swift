// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "latest"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "5ab69622a97cfc220484366cf280aebd2053ecac58fd368f9c727947e33a794e",
    "sha256" + debug: "ce57749f3b24f51abd3f0db1fa11c1d89eef824e2cb2dce36a0a901f0dfe2845",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2156bc924e0737bd64ae9a046891caa0f227f4dd02e25c9adf382d4960a83d29",
    "sha256" + debug: "2f345c3ff50499ba60397fe595cdcb4bf2834423b4bd8bf7ee01a416fc314713",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "56bdcc813281b841747dd4eddd7cd9e2fe5007461f719140189a3e3d34062c8e",
    "sha256" + debug: "d555f3e8e314ef9e2a4489080487f2c579eb0190465ef8bc1e4383aaed0ce220",
  ],
  "executorch": [
    "sha256": "2e6178d323e3f33fa30d7c15e6073a6b8b97e42a627ae58bf156693dcea127ee",
    "sha256" + debug: "84c63726ce943c82e3876bc1f467f82801cdce907d29376df89f888bce473fe3",
  ],
  "kernels_custom": [
    "sha256": "823cacdca61d4b5dcfbeaf02c7bd61a58cf0ff008e7d948852df687e48bc8c7e",
    "sha256" + debug: "604b5e90284cd089408d2e676d9d4c799390631363eae2f04b13a4c4a7728371",
  ],
  "kernels_optimized": [
    "sha256": "d2602ec7b49a33dfb8c34a63b2d42b446e878cc4870ffbae9707a6155e85d340",
    "sha256" + debug: "6f95d736dbf8c3e999cede05c72b320547f3332999f8fbb601fdfd408d0b5e93",
  ],
  "kernels_portable": [
    "sha256": "2174a0641343bcc8f2e9435c29937e0ad5801ce7dba40eeef74aaaf3722270b9",
    "sha256" + debug: "f325f280f73d6dd0859cccd945281cea87b2b8d0d3526f090d1f8d95515f891c",
  ],
  "kernels_quantized": [
    "sha256": "2272d8d478dc5b982196df8b1d261775d90a59f1d209bff6c390eb42360a0a15",
    "sha256" + debug: "c0db5c5f63aa2ddea6d0939fcd7450468a8b391f7d195322bc79c2d16e7566fd",
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
