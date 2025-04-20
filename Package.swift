// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250420"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "e8599fa983da011c8143b53d6ddb35a315724ada076899a9c98de31a46e3b44a",
    "sha256" + debug: "e3d21c57ae53e0b341d496849c251af4487d80883a3c8584fa3171154b8220a0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "488c1d5b0631614ee76fed00e43213d6c96ec9685a948173bda73258e89023ae",
    "sha256" + debug: "a10cdf227e2404d1a07cbadc279e5a9004a45d6fba942f534ae33195594bedb5",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c4d5c99e8292b7433c8dc647715bcdcc30a4072e61f2341354654671ec717d48",
    "sha256" + debug: "627c3d0f667583864118c37774c1554d73b8a4615ccc8ed5811bbca884220177",
  ],
  "executorch": [
    "sha256": "5b32fc723f26dddbf57085e43653fd4a329666f28dbc71eb3e39afcdc0101630",
    "sha256" + debug: "c55b794652b07891636b10f5f42cc48f0bf4bcdc652104916e7e6083e0c7390b",
  ],
  "kernels_custom": [
    "sha256": "215b69bfb558119eb739362431c30bbe3d56c3d7ba48aadc977ddc8d74c500e8",
    "sha256" + debug: "bf2cfbd51e2e5c6fe4c92605068e875ee9d53dbd727de9f67fb52e8491ba8a19",
  ],
  "kernels_optimized": [
    "sha256": "d7a0cc295cf36bf46c27d1d033f926af4b6c91fa562db51e09d53798da6512c0",
    "sha256" + debug: "13e63cca60bf17aeff421b84af234df242658f23dcf551287e1f57f515b5783e",
  ],
  "kernels_portable": [
    "sha256": "e4ff2dd6cdb7be0021b3218a58e25ebb6493479e731bcf6aeabc2728586942a4",
    "sha256" + debug: "c52292aa0e350a32913dd6289709cc8256c285e65d495772ad9eaa097332099c",
  ],
  "kernels_quantized": [
    "sha256": "2fea3242d3c7cbde4ab8fe1c6d81394fb2fb148afb23b786fa451c63e1c5ad9c",
    "sha256" + debug: "8f0abb2bc09e104b382e80bad29bdf42ecccbfa7234925e859064c843118d0db",
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
