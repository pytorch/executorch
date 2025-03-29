// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250329"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "16210955bf2242199df17cb4455211f4bf70ca8cb87a6e7d7daf13f37caefccf",
    "sha256" + debug: "eacd0e8d23874a47c12296dc8bff0c41d78d1cbd3a1529dcb9a91605f1d37c23",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7fdcaf090c647e3f129d924e7fc06d7bf2c7227f2a714958e80e5d88718c2380",
    "sha256" + debug: "030f61529709d9d58f3f86e90653b8d69644a1743c83809f74e737fb0e3f0bc5",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a9ace139cfcaa15c60e718776437fe6da1062043b156d51a375b9186c764b167",
    "sha256" + debug: "8713fa74bcae4d5457020b91df3a9e623d2f3469c2cb34056e8e47b33dcffe2c",
  ],
  "executorch": [
    "sha256": "f68e55f23561e76de6fd6adfd3c1a0f1700ba87034aff6d446104330e42dbd17",
    "sha256" + debug: "5b362f71e81196f1a5fc98db64b9e390d1a62e9327d25cbf0449dff91cf31033",
  ],
  "kernels_custom": [
    "sha256": "e756a59fe2b92c9ce8ebbb1ba2fcfb82dea33b284d6719695eb166497ca4d95a",
    "sha256" + debug: "300028d1edf53378d68e2602436201f25d3b7189ff5cade37dcd5b4e746ba364",
  ],
  "kernels_optimized": [
    "sha256": "11a029dd38ad0d5c4893d2731f3b72146f89149432b6af6ee798be9d214ab9ed",
    "sha256" + debug: "46572a18939cf746be773e18c20b823f8577a1badb382d2d2b7532573f5a9daa",
  ],
  "kernels_portable": [
    "sha256": "c8e640418de2e58b48779c1025fedb73dd2fa3d6ae4216efec4825e9f7eca8ca",
    "sha256" + debug: "1c8640ea4d575e15401597e8a4ae63b6bb40d28eed9c19bcefeeca5846f43bc7",
  ],
  "kernels_quantized": [
    "sha256": "64906138515f0fa8720f3d5b50c7e24b4ac39c41df2c87d8015e75bdc2046424",
    "sha256" + debug: "e4ad0034b8af511b7fcc0b5cd48f31b25226c64f915f8a328237336fc80dcf5d",
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
        path: ".Package.swift/\(key)",
        linkerSettings:
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
