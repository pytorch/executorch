// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241201"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "3fe27b91339bd1872c08772ac883ee5f4ec869e1bb9dd5a51c3950b6f781f667",
    "sha256" + debug: "8832de62ed5ed40baf090cd12adfe6dcc8b8114f36f02b517fc86d33862b24ca",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6f0d99076151c37fec3b8e12d6133e59c4c1759c4e1391cca5fb7f638fb29b3b",
    "sha256" + debug: "bc60217e1b7a5737aeb20ff39c0966870a491b629087dfbfa33d7a2c52ca3f63",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5677f2969e708e3c729a9e9ca52790f5f9376451d9217a041f21fe4ffa4032f5",
    "sha256" + debug: "e8b32ab65401ba5210d7ae4e45fa53763ff6a0c0291ff952901e93966825051d",
  ],
  "executorch": [
    "sha256": "19d7687622ae7af6c069767969a3976d02612552b759edc4f43cb212e4f01a1e",
    "sha256" + debug: "a525f994f852860243eaf6409fc6adcc089ba3f355b3828dafe420611a666a75",
  ],
  "kernels_custom": [
    "sha256": "de0e1af8a89d0c7860f0f0e19eaef97d6a5c3b7bc0a10c7fee0235628d66f069",
    "sha256" + debug: "965880880bf8653bfd07607221c399ad647f549a3c320dc3abb50e2bf270ede9",
  ],
  "kernels_optimized": [
    "sha256": "712bdff292f646cfc74e48df982338a8e2efa5d11dfdcbc1e5cf1915ec86c23e",
    "sha256" + debug: "4030f6ca5454b0fbf2b26fba27a4d24e54e4c08da2a23fe02ea26cb62eaaacc6",
  ],
  "kernels_portable": [
    "sha256": "1df18e43791d88624c5160d4094b0fc6f73838efd8fe44819ca4f5e8f3f3392a",
    "sha256" + debug: "78f780d6eb0590d0c9cbbdeb23a7853ac8e7f49582ae68ac14992c8c8dcf2292",
  ],
  "kernels_quantized": [
    "sha256": "2472d09da0ee9771df48a40d7be7ffc6dc003df6853b8c1789f72e1b01cbc977",
    "sha256" + debug: "6234201c8bc3e606461a27bd2da22d435ddddb0df474731e0f06ccb4216467d1",
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
