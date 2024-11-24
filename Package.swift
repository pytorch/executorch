// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241124"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "4879d4f65c5401bec202bebb2feaf21882f74bb981961acc134ef2494db1f6cd",
    "sha256" + debug: "b45e633f0e1d4dc211f6ed3858bb2ba1efc4b5fc13036536cd8de7dfbba94fdf",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "251160744ac30ef7a3eb2e32abba520d84f1bc8b660b047639b2d13dff6e8848",
    "sha256" + debug: "0bf798ce47ba78ca0ea43c151a1c09e5aca3e857889f538edd4b16611553f3ae",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c6277e2001f6d050277f1758ed940e1e1d0233b209fe7ec1e7060801e6341003",
    "sha256" + debug: "6fe9318420d6b604f0a6be28744d4f4fa241547a100c62c715119b19a6156927",
  ],
  "executorch": [
    "sha256": "aa694b27c5dcfe3d8f4499a53263c2745ce6571f259fa2e23d845766ea286280",
    "sha256" + debug: "b7f491fb43f14c25329a80e07a96f135e48944d00c39d3621c7f827432a5f363",
  ],
  "kernels_custom": [
    "sha256": "70aaf14eb3ec9abe2d151d7125abe499411fe673c87e88e5bd42e2e14d56e6d3",
    "sha256" + debug: "95dadfdeed3312f3cfdbf6602e0c8ca2797251eee09e20287159f9d1c5d3dabf",
  ],
  "kernels_optimized": [
    "sha256": "c01a0bc7baaafd383ea6863a92592e68938df67bf4166d5f9386f1388cbf715f",
    "sha256" + debug: "2d743d5a5ea94eb80018275b9338c63f1376213ecd251b369b77100152058502",
  ],
  "kernels_portable": [
    "sha256": "440c058262c690f5e8054ab24f0370893bbffcc19fa072766a572df9c0158ef5",
    "sha256" + debug: "ec9db42c5cbdf62193eb3efd02e195d3586613d3699c706902c5b64a83e77ca8",
  ],
  "kernels_quantized": [
    "sha256": "48fe6a2a17a91b51ad3b0fcb46305d7fa4cff4e7368df2e46f768f8f42f469e4",
    "sha256" + debug: "eb58dcb5f4cae53d84d50b9074048556b29b05053f265ca4acee31ec407574d6",
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
