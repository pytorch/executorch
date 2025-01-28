// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "2cfd518d5f28cc7194c873eebaffd61631b8a30d93acd71aad1c71e466eac55d",
    "sha256" + debug: "85bea6473f896b112e52762108420372472640430af7b3c8315f5161008acd2c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e208ed8cc5e24a71391e8b85c59fb7d8daf75038a917debac4b22948fa8cec4a",
    "sha256" + debug: "bcb15933efa4f44b39a7d1cac04325586dfb8ad2ab7f8ecc9262ea224e0ae328",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e4ad94868bc7bd96c4e03440599d41b4c9153e614130607d8c7681064dc1742f",
    "sha256" + debug: "ec16bc128a03ed8f3498f36b7d424be7bec6c95bc53ff701559459d3c8943c8e",
  ],
  "executorch": [
    "sha256": "d929fe3ecfb72f24d9f407df76a5063de684b6081658e13f2f7e653b46f098d6",
    "sha256" + debug: "097cbb220d542e85330b5af9822af6dd1ece587e10298694cb2eaeb029545821",
  ],
  "kernels_custom": [
    "sha256": "5cd05ca7f8c4ade262741ea685026be3a98e2796be8eaa56ba14c7298f91cd17",
    "sha256" + debug: "a6f9e8344db4532a5c9e24de5a8fc3200d1f83add6d11b92705987f8273055ff",
  ],
  "kernels_optimized": [
    "sha256": "5207cdbcb541576ce29d0104d6e83c37e1c0beb51b33c0aec9a48d0b4fd14857",
    "sha256" + debug: "cc3d7ec95504df4c6f4560dce86569d0993d9096e7e29d716206bd4683286344",
  ],
  "kernels_portable": [
    "sha256": "1d1d84b8f2b44ec43c60e5d177d4eb102a128817d1afa079aa96d4499a02f249",
    "sha256" + debug: "69b7646980ce3e870b7e0192a187370e1c53481ac2dac01f3e5b67438e36f6b1",
  ],
  "kernels_quantized": [
    "sha256": "60eeaaff0a3375595dce0b3201e608bb54f834439308296cf094e5858ca7d1bb",
    "sha256" + debug: "c667e4643c2d1b10323f13cbce51ed26539a2ee9d71a99e0009d355d5ed0d9df",
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
