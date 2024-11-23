// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241123"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "c2ade7d2e7b78919ccbf157e8c87d84c5d2e7d5a88e2df9b609fb6610caa59c4",
    "sha256" + debug: "9437b2106f20b4337b4434a6af7ca56ee7140630a7ba1b5d8721c4b1d7441103",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ba246be44c35dcd902fb6233a80483ff8275042ef147526144ac6d85ffbd8ef9",
    "sha256" + debug: "a52eefeec170d401b67210a5b07c1d76fce818343f28b854825d571d4948e087",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3fdfa7a13dfdf4b4e67f6e1c3f64b29158fcdc025c9ffd792e8f8757577949e4",
    "sha256" + debug: "855fe3261d3922ae0d4c436b68e42b153963e8a4a54073855bf9e9d4231daaaa",
  ],
  "executorch": [
    "sha256": "5408a00a030b291435d32d9e34c0c1fab42fe95b1609f1e69e60c3793216d27e",
    "sha256" + debug: "40617b14c40974cbab0240e5cd8bdfeefef8b040620c3d82f0e95fb671963d36",
  ],
  "kernels_custom": [
    "sha256": "f5ed04c6b7268bed159b86f1b484eb058e66134eaca2fa18304d56dbd15d6d22",
    "sha256" + debug: "0ee0403929a8a7def15f8f123ace79cf382cc0a3f893b42f4cf7ac02032e4e53",
  ],
  "kernels_optimized": [
    "sha256": "5d8fcbd24e064912df827d0a33b4e3e70a0b84366ad5700fb83497e4a9bf73d3",
    "sha256" + debug: "1a106ec7ed634d9fa3bc125ecd042b1915c45976633d4683d6b0b1f1d829755e",
  ],
  "kernels_portable": [
    "sha256": "70a05dc8c4cb159de1d953389c481f24af056589152612a5cf5f748a2f3a285f",
    "sha256" + debug: "2d6ce59de19d89829e257faa0aea9c6acd9975f027bbab431c3a4f763fc86b14",
  ],
  "kernels_quantized": [
    "sha256": "1fcc07da330b2de71c10fe50316aeea6a441893d775d869d293095b7ba9fa290",
    "sha256" + debug: "a450616b7213fb1e985053f340a57a59ac0e9eb09e78803e791997a835384501",
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
