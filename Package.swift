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
    "sha256": "c2bc52586ad7772555658cc8717068c12b79e22aae69c95e70b29e9578bf290f",
    "sha256" + debug: "37b4b649ef1372a8092d203f84916989150ee040b7fd70c2c3a08262e05f0a1c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3c2ce1b04c7476d2fad0d0a3c27564f4a8faae46af5f2b371207442182d75f3e",
    "sha256" + debug: "045c05530faa70c7e1030a9c1ffa4ec1f2a90dcc55b0fdbe5fa409aa09d0f834",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2a7be4d56cc576b4adf28955e635808cf933a038e05a4d3778eac1df7c97cd1f",
    "sha256" + debug: "71ffb85f56aaf87614890dbf9c6ada71804dfecc6454ada645d543699a088dc9",
  ],
  "executorch": [
    "sha256": "b41a2ea6a2508ef0bfd7cedfb626c8c5fcf3d4e63c17bc6454fe1c9979c61289",
    "sha256" + debug: "1e7b46965279f1f18a17a6ef28de0aa9ab58984fcef796412f8c1250994aefcb",
  ],
  "kernels_custom": [
    "sha256": "6e671bbe23e35d6afd1f878a3c0b6cfe364c66adec1492ef075591259c43f9fa",
    "sha256" + debug: "5d8e202879dc4746e2daea7d36f8ad7d63516f528ab8eda776cd863a2039a664",
  ],
  "kernels_optimized": [
    "sha256": "ec1802fa56742f947ba5e8ec0a3d8e97862db86cc2d05f07bfd866f6ae9fa080",
    "sha256" + debug: "3c947aac2a274721a5f98c5b5b4e9cc36e855ca767c798efdf4483c6bf9b2e46",
  ],
  "kernels_portable": [
    "sha256": "5d40cd13e9940fbf27146e45b9300b3c2762247b94957daa485a3f8ba73c23ca",
    "sha256" + debug: "2a2c7bb9c92bc1b8f35daccaf8169d0d759964daf95ebecbdde97b9691263ac1",
  ],
  "kernels_quantized": [
    "sha256": "ec7f49a2548b766f8440f70d918b7edc66cb8ae1fdf3626eb2e3b16555233243",
    "sha256" + debug: "a4ea66f859f58b45c41dd9092b55e3c825310d7aefba985f3ecebe4767b486b2",
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
