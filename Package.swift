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
    "sha256": "0defe2c8a0142533a5f5584aee1a80f01ff31be0e2760f9032a4d8a53489ab72",
    "sha256" + debug: "1e5234a6f9b87201282762e670b0eb4d607afca878efd447038945e6265dc02e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "53e5300f8e9937bb649dd9430d723ac133d2ef776a3c673b832ea5fb6cbc5d86",
    "sha256" + debug: "8ff63b857a206c4138c3c99efb113038ae7e42fd4dc4f1d11db8f6260b21522d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f324e97dc4883d1e076933b410b7c0cf1b1e9e099d66da9165a149e603d00bb8",
    "sha256" + debug: "46acc09eaa3e327240e9617d3b2260bd1984530450bca4c34e4adcaa5ae34bb3",
  ],
  "executorch": [
    "sha256": "644e804e4418b4d5f9ec39823c633f8ee254d2757af16317905f9a7e183769db",
    "sha256" + debug: "d82540aeedeb0009cbb8f8cbd7149ccd9e2c657b755a2292aaf404c471fcf943",
  ],
  "kernels_custom": [
    "sha256": "6fb66533b14fcae51041b42119eb663dcc1a9c17b9ca70286445724f2ad99b12",
    "sha256" + debug: "253634f9b788ee7a1ad7a087dbf64aa9104607baa0bdb1bd46e9843948c2801c",
  ],
  "kernels_optimized": [
    "sha256": "d0ae6f7d019f35014137cd9dbf2c38100f05bd1eeab9f1ef07e51e53219e819b",
    "sha256" + debug: "bf18f6c21bc653ffb29b4259443900169d4659ecbe9c4622b59f36c086be00f5",
  ],
  "kernels_portable": [
    "sha256": "01bcd64548ad2d092ce0a3a3fde4feee702bc38c361ad8f1650a598a1b063b1c",
    "sha256" + debug: "901cd876719531ac8cdf96235f304d36dc05834425cf3434a8f124b5bd0e41f7",
  ],
  "kernels_quantized": [
    "sha256": "cfc1b3739f5dd201e8585966ad31bf1711299737e89906a7f60718512272d2b2",
    "sha256" + debug: "65227b985b7729487e4e28778c810fb33bf3aa83eae89a246a528cf046b7e2b3",
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
