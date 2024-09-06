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
    "sha256": "cb8d70de2ad202c21670397dd320e34f3d92747f8ac8073572b8a173d2f76bfd",
    "sha256" + debug: "67e2d3ef59507098df363f4e44635012221eccc12acce7ffbf9d630c58f2cbff",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4dec1d4a088c86d7c2368a10f63f0215775672768b12e4997ee2556a853ee680",
    "sha256" + debug: "1f326ecbaf95a083bd0e766dcf59771f915926740b47cc328b690824c312cf73",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "20dd4f1c8707f78d1d36100bd9b28a0731a14e3c76b6137bc21c8c8e1471bd0d",
    "sha256" + debug: "6d9f1c51862dfc59e43eedf1d3302fb9aadd2e10be2c52f8e7d0ee258b9bc73b",
  ],
  "executorch": [
    "sha256": "78c33c8b96a24b64affd732aad3e94dcb0ebb2b6735abd5dd1656d36fe791279",
    "sha256" + debug: "167e5e8de054b57bed57a31ce3e2eb7852f8868b3fe0420e55d9400b814116b4",
  ],
  "kernels_custom": [
    "sha256": "d665951d3f5b0a14625eda21ed2068212669414d76a1f4a8f62fb5052fbc5fb0",
    "sha256" + debug: "df427e974a49d3fed8eab505e0488ba91f37ac8cd24cd80686b62d3782529f3d",
  ],
  "kernels_optimized": [
    "sha256": "969b655e153b03ab1f112540ea88d5b96620ce265084207992e51a6189def94c",
    "sha256" + debug: "26722f7aee072f928371e74cb26a96d9b96bbb8957f31fb4b24c0f283a85403c",
  ],
  "kernels_portable": [
    "sha256": "6c76af0cd2cbceb36552dc7e93b233453d2834edb9ec5fc130f989ff577c226b",
    "sha256" + debug: "db1f1b912dbc74d6bdb5696a3190c9bde7b0efb50f48f940aebb6cebf0941b8e",
  ],
  "kernels_quantized": [
    "sha256": "cffc38db480de9446ea6b238219dfdc5f7193e063ad8253270f2ffec07cb32e0",
    "sha256" + debug: "334c491cbb073d8c81982e97769644009ed60f393d0223407048c1f421a979c7",
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
