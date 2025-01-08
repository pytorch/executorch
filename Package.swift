// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250108"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "2ca815e5f0ff760e9affccfd79474bf2014c15df6b20f680187713dc4295785b",
    "sha256" + debug: "a91053ca8a7b3d01bd64dfd7d6ac58c71529fabc589005ada2757064c473c9fe",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "bfedd0615bc4393087f2ce80582b7cd0f59f57d3a1a3a0f487ae9b0ffe05b01c",
    "sha256" + debug: "47dd0e525a2948ac6086ee44ffa8841afcd57bd397b66da59907883dd92e08a2",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "060c02f8c1790af9741adafc93614552715e47315316ca9ec5b7c16d0dde3bd7",
    "sha256" + debug: "c860edb6551b566c9fde90fdcd175bf8802ea9c37236a4ff99c6f336e2758a8a",
  ],
  "executorch": [
    "sha256": "f096f0e15211a0148a0bc628f1f818a03a6b82d0564ccb3441063a60652a7f62",
    "sha256" + debug: "e9b3b5a69746d123b2e360794a35bd95ab48fa9f45161edace64f11353204942",
  ],
  "kernels_custom": [
    "sha256": "8e69d437522f0132bc8f34b8c50672fdb022694568ac0d53229aef3a3907b1b7",
    "sha256" + debug: "e035df5a150710ef97fe2cf9fcd6200765cd375f4602b76d5c13542742f8cfa4",
  ],
  "kernels_optimized": [
    "sha256": "c79a7baecf645902dd33ed4a3690105594ef438d2408e543db4d44762bf51d73",
    "sha256" + debug: "e63f76e0326a98c879e0b6557127921556d407765280af5e5a93734a27d1cbc3",
  ],
  "kernels_portable": [
    "sha256": "87834df12ebe5b56d072cfc8b4b72d37b7b36952fbe5792bebbb98f9640807b5",
    "sha256" + debug: "097abfe909ccdb4a24b82405b23ce376307af57d2d445b1a6a90d7febbc39328",
  ],
  "kernels_quantized": [
    "sha256": "c7c8b58adad2f90bbf2b2f68fc7e98a5720585bdf4acc7c870fdb8aef08d1680",
    "sha256" + debug: "617ba87ad34f3283a4825a98830dc6dd40159fd8c60a9c90f375054576bd0a34",
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
