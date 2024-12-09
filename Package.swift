// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241209"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "b8c17df03b698b7ef3c1770945009384ff9893bdadf77b0fe6052acc661a9a08",
    "sha256" + debug: "0f4241cd8bb8b9a2cd89ecadbee1da314a6459751bae6893f8c8a58e449145c0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "958c41c2d3b6e6fdf817846548d268ca04ef55129a8c4874db96f7063a72484b",
    "sha256" + debug: "bbfde274d12e924c27a44435cbc622616afd0adfeb9347897b47f1b838c9c7ee",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "dd838dbf23b8aaf24066d0505261f7a8472561171a3eade6ac4d64889c5edccd",
    "sha256" + debug: "6258f44875854f353c8da0bf43077ed8f156f963cf661e0b32bfb31a1bedb418",
  ],
  "executorch": [
    "sha256": "fbda2e75bd4b5dfd5ee41ba940e865d228440ec62668a433f29769a451c098eb",
    "sha256" + debug: "97bd7eb36bff6eb3ab2518d99d738eca8b644875fe643bfd7de80bdf14eddde6",
  ],
  "kernels_custom": [
    "sha256": "063593c48d8688a34695b7845ae7109d1f8cc44dd2377754490b5aedfccd07b3",
    "sha256" + debug: "37ea8e50899b04b3f171d4bad395a40dd4d34f82ed8cf64277780f0db17e9e2d",
  ],
  "kernels_optimized": [
    "sha256": "b3ae103ef34d324c731cb5bb6f71eef3d8c88abedc1b3596e2f2db58590e48cf",
    "sha256" + debug: "98cffcef1bcf80c22157ef9d5f31e4ab6679a524ae1a302d5aa5358bf0450d95",
  ],
  "kernels_portable": [
    "sha256": "78f85b9b7fa1ea9e74b3aa04317458d33d05dd90b754c85a7743b0a596ab168f",
    "sha256" + debug: "027b6301a97fcc0e27c0763bc06d02cb3980fe5ac6c80dafed31b5a6a43e0037",
  ],
  "kernels_quantized": [
    "sha256": "3613fcebc2155cf646b1b04e1b8767d891e505b2be83dff222523d4c9924daa1",
    "sha256" + debug: "8e9eaf22208ae9ab2fad5c5fa9eaea25165765373042d3f7259e1f09597bd8e3",
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
