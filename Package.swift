// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.3.0"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "f439e8480acb10a9a4c7a90973a78dee1d214be7b12e82cc03a97f07254a6928",
    "sha256" + debug: "d709244333f9874dcd5ba5d571f7b56aa295f667b6e2e4496893ce956ddc877f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "210723c4ef6b9969e5b1ba5f493f2a8283809c5b03a2c478c89e0b82fafa0b6b",
    "sha256" + debug: "d0fc350c5b99617fb62d0c1e2a44b8c1ab9cc2b34a4b69e4f679a35ba2082625",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4deecaf68b7d9840af9359717b6deaafb34e1f50353434cae87ce93ba5e02e00",
    "sha256" + debug: "f269200c03075016bdfa306e25c8817beae55decc318ef345a3e2fcd938e2ffa",
  ],
  "executorch": [
    "sha256": "a8348d514aba377bc0f8412f41817a561db7a23eae6d699f2b592163ac3aa721",
    "sha256" + debug: "4642fa3e3d3893a8312331ae5e102de01a38edefbd68cbcfcbdac721a41fb322",
  ],
  "kernels_custom": [
    "sha256": "803b1c09b2d736211baef620c844591a03c1349ddd989133b5090bc82e4a1576",
    "sha256" + debug: "fa7b4acc12069a63f7bee8cb905d49e872e9280285c183b1a9da75fece946a81",
  ],
  "kernels_optimized": [
    "sha256": "efc72c1073c8d99169bff1b81d1c3537886dcfd220a36536d5f36e534fc0d029",
    "sha256" + debug: "cb567a8ff1a853fae7a4b4fd4de31664244ec3898064958255b292878c08be15",
  ],
  "kernels_portable": [
    "sha256": "1ef1c14afbbb09fcbc659dfc3a0f0f9f47a5b1e19384a930fed5b793423ee79e",
    "sha256" + debug: "938cd12f07551e87ac7b5e170285f2e8eff42bd3dc091b49a2e53647b3487840",
  ],
  "kernels_quantized": [
    "sha256": "1e21385eb266225cb2008589e630b2cb3fea930e67da91cde7f44d3bdbab5d9c",
    "sha256" + debug: "0bbab32bf1cea3836e5012efd652f616d22426e247bf3c485c4872706a993673",
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
