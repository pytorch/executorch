// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "cef3f8c6df1fd0d463b38fea36f4167f8d995a40a8e44c1cd5bf5291145e973a",
    "sha256" + debug: "43a5196b65458c23f510ccc04e22345f663ac29b915e1b20c6a8239aa29f55ea",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "fb620a18acf07c44a5b959278ef3c5b74eace7516b3c0e6e1cb841d51986745c",
    "sha256" + debug: "78f6115f5550dac06fa38fe09fba6e761685949f47b94fb9b4cbd83710d1ea99",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b9c6e5426aabb0806db28315e11abef52e2887feb942adfb71841c24824a4e8e",
    "sha256" + debug: "ed1727ff7d9caaff3889279b58588d85e640d29cbbe10e8b5f3a926950009123",
  ],
  "executorch": [
    "sha256": "7b09f8242c944e92aa0e634cd64a1354af103ff69a9106190704050a176774ee",
    "sha256" + debug: "95dde09176cd65e1b7dbc66f82be3d56f6c5e4509bb15c6556663fe69c4d5c4d",
  ],
  "kernels_custom": [
    "sha256": "6bc7107d4074ebcd2b541ca06b2911d8c7eb493e8c20dfd558ef4bcef6878c94",
    "sha256" + debug: "34643cdc751038ac0776a97501d5dda3724fa7923530b1403ee78f6d6cbbcec7",
  ],
  "kernels_optimized": [
    "sha256": "9a5b06f2d88f7f27262f2fee6e8ce48f453d21e0820d0b38128f85478af94101",
    "sha256" + debug: "9f12be94d4b8771e395e6abd531ab29b3f5e734cd7415b6db17e245b35c7606d",
  ],
  "kernels_portable": [
    "sha256": "0fcc32c787f7b4bd8e5126259a1970eee961ef14547909132338a9cab7bbd1aa",
    "sha256" + debug: "069ec16a3b37bc3db516dbadddabab158afe9e83b9122a315b9f71c8b53eca67",
  ],
  "kernels_quantized": [
    "sha256": "1657c50ca25dfc2f8260b4877f9f4dbf0c9e1bbde24e61a45befc48f7e710914",
    "sha256" + debug: "3d7d919c06b86869aebc8113963d3f308f617ab2867bb3e838c404c504245eac",
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
