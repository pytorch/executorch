// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250112"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "4d6e0128aa855694a6200160506656515af56c55cdfc9b1e59432a0a650ffcfc",
    "sha256" + debug: "8241df4bfcb5570e9473fbddbb49a641fefd8a9509c7eaf492eb23eff068d96f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "46730fb37df21ac55cab97bd168f4e4325d5dd54fe975b972aacd885308d1353",
    "sha256" + debug: "fa1cc0b40e2d1af7c8dd7af6e1bf424f58a089e3a8359f5015c05b50b053b1f7",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "42fb43c7e135ac0d5fc25f246fff4baef64e01073e11a1188fcf9f2b30bf572c",
    "sha256" + debug: "65d610af0e6a0c0128b07b5c7b4cb17ae8b67857fd3b4389d1a6c6234da6d127",
  ],
  "executorch": [
    "sha256": "87cacdeca34e9d2373dd4f77b943de34df9b6690137248bd2a2dd95a381145a2",
    "sha256" + debug: "7e0610a6527aae006e64bd10399af8866f986be4e964d573aefb769e0f62f22d",
  ],
  "kernels_custom": [
    "sha256": "3bda97a94f10130296220421cec8d2c2947a41819a5400d861890e9f24f3dfe7",
    "sha256" + debug: "e86617cf17b0538dfb100732f395847e02a455b554aa4954b04f54893b239701",
  ],
  "kernels_optimized": [
    "sha256": "5b01104003b2bd8ae6e80d50ae7c98ee3ec2b45bfa76c5c02e324e8eda4c15c0",
    "sha256" + debug: "e13b2172e3dc186b9586a7d54502e606a9586da235f910d97d13ce453bae31a4",
  ],
  "kernels_portable": [
    "sha256": "3ba192013435766669ccd4abe5ae32805a16809a8a6ea0277c742bd7debdec54",
    "sha256" + debug: "ff1d19a4ecfff6dbdf261f7794f755371ed5b22638a29ceefcfbccce702c368e",
  ],
  "kernels_quantized": [
    "sha256": "42b53f46e5e4eb0a3c20cac97b1f56ca88b23b56444b4ecc0336b30ad175958b",
    "sha256" + debug: "42d53c7039b9c466037e76e02b9a50c2d076349f4d2109fc16fe7dbd4f29345f",
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
