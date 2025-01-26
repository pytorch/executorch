// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250126"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "f6ab9202efc3504fdb681beb71ec4cbb44132524400e66a0c57440e5a1843a9f",
    "sha256" + debug: "6fc931f9d68248f452112923507207fe61017830b40280d3298d818386e5ddcd",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "75f2e666aea6adde62045ebd82ee63fc35753525de344fee15c8c9edb4dcdb55",
    "sha256" + debug: "11727ff8aa02d488adf42f15eb56748c65fd9e07420fcef747749e12ee944365",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b400a766d336be0e5c3fba5f575cf2b8102ab81135c8222c5e6ab98bfa568bb5",
    "sha256" + debug: "9f640ea6c73e2770cf91f1e030b56d81b8b86c6d05992a2fbc190b4343f07b61",
  ],
  "executorch": [
    "sha256": "f7aecd8ab3745e0aed1b1010945f365ccfe89c56af5ed92e32d273f694ddd6cd",
    "sha256" + debug: "86e0d4a873db690c1067bb9f69b918e87e6a9d7ece2a53b19979f0794c01d508",
  ],
  "kernels_custom": [
    "sha256": "ff4c3db1c12e6315f8d4cb1ff6e039f44de47be5e1a442406d2d53ba01718583",
    "sha256" + debug: "a4a5d253af17e2dc387aafe5849d406bcc9db5583f6d99bf3bfe23b8e52dfc7a",
  ],
  "kernels_optimized": [
    "sha256": "720906e08d2c734baf557e09985b89f662aa3f943838360653a45e32cadffa27",
    "sha256" + debug: "5c2ee41a27633f507cc73b66e474cc95312dae78f0e0cc8ede6a5fcac9c7c93e",
  ],
  "kernels_portable": [
    "sha256": "3e3639b5faa3811a2af7ef99b5e178be1101b691015fb7bb50be4221bf29e6f3",
    "sha256" + debug: "b87775ff344e4b4f3b6e5e390188ee7488b437413499f6de3725cc0bf65e6269",
  ],
  "kernels_quantized": [
    "sha256": "81ff6595632847b279fbefc541ce738a4c1959812c0a3c6c7e426e8d581222a7",
    "sha256" + debug: "ec770161860a9abc19c794fbfdf4c90eac2bc77a50bf31ceb41dc832fb978e82",
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
