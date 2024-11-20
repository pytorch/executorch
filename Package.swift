// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241120"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "b8dc86f3d32ee37564bc2ec380987d384d6045a9adaf0000f74c14a3c534ec22",
    "sha256" + debug: "08d280fb583ca99d4ef379e80f4458c90bb76b2a94df7d87b8e840e2c317dc2b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "479d0ab74308dc2bab48e2f24a7a47246b1c53fa0e51b486c9d0e30c3349c821",
    "sha256" + debug: "34148ae5de6d5cf68201df834c06361c9f3e12ee003b8017627f9e8445828a8c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "edec28f79f6283b997ebd1da3475e7aa465d53d17b429107afce7b120a9e49cd",
    "sha256" + debug: "b66eba3ed4dd5e2ae55a811c8a7542101ae481e4243d11a451dc1c374adbdaa9",
  ],
  "executorch": [
    "sha256": "8be021332c990a6a054d665516d7e9bf4db1e324c7cf3d9b3695fc1ce8fba073",
    "sha256" + debug: "8cf3c9a129d6f884bfa4ac693b645ed5dcb6de2b2d2fb3bc7e23abc0b45d8a3e",
  ],
  "kernels_custom": [
    "sha256": "0763d0d3998e10d77df6becf4d0e4b70e72bddfa77f0006e6f414b6360201b31",
    "sha256" + debug: "1366628f60863e97d2d3c22d75690bc99ad687cbf9e4f7ed9a23d3834ee45119",
  ],
  "kernels_optimized": [
    "sha256": "e46fc2606b42fcf045efbbbceb39ed7fb5634b0d784a41522a0ac84ff89e824b",
    "sha256" + debug: "b4429ab6241dc6ee0052f880a9c298458d50f41ceabe18f1b616962ad2132510",
  ],
  "kernels_portable": [
    "sha256": "8bb0aae66daff81722a0b5d65036c2726609377a704fcc13d43860e8a6fa9874",
    "sha256" + debug: "e29a811b752540862ffdd9eaff9512e076e0ebf24399cbdc83343acc6af3755b",
  ],
  "kernels_quantized": [
    "sha256": "a4c0524fbf8be01578bcebd919b5fa0994d514ac10165d462450b007fe037ad1",
    "sha256" + debug: "1ccf8a8c9901072652ff6b42b90dd7d92a39aea7cbb8b36231158085b2a8fbf0",
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
