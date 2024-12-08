// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241208"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "320cb3b9fad3b2de8b259c1eff2c03f605235140726c0d737581c4719f9cb83b",
    "sha256" + debug: "77e40368aa34721639c179d39a2cf441835a76d9c55816d30c04de0907e9bf2a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "566d30aa67174c81e6b03bb718805ff3ea526ae2ae2eaa84b742310a8e1a6729",
    "sha256" + debug: "7ffd2e0043645c2786bd860ffe6d463852b98f08111399784c1273943c5a4ffa",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "12e1ebbf1226de68049248d5281eb12f45c29417ae487b10ca46ef5b4ae58734",
    "sha256" + debug: "b11ebb3b288faea4e6c59ff58ff0677ad1eea5ce9c8152614b2b872d59ab637c",
  ],
  "executorch": [
    "sha256": "c6227acd86e1eb782f4895379fc5d4f515dca373178c65c7314149985ba74bb6",
    "sha256" + debug: "5cb9048994eb1f41a6c13d4e07f706d670e583503428c89016186ad113635ae3",
  ],
  "kernels_custom": [
    "sha256": "268901d260a1793d0a678ceabc85d1b22294ffda6da8c27c021790f2d78a830e",
    "sha256" + debug: "61f175f064f4614962af8d4eeb466b5bdd2c525f86cb36e7ade3ec538193c2eb",
  ],
  "kernels_optimized": [
    "sha256": "8dd38735dd5356916a12f32331478a0df8418a85ec7c4ef839395af3c4cbcd62",
    "sha256" + debug: "0ccc1d0995207b16fbf68b825fc44649f1842b258d412fb2b7f90c82a56008f3",
  ],
  "kernels_portable": [
    "sha256": "f2030023038159fd13f021d7f84df82de68fc342085ca0ac96ccf497b3253646",
    "sha256" + debug: "bdc6c10138e3d7ebe5a04c7c300454519d2d849a10a17d7eb9cade4ed9598173",
  ],
  "kernels_quantized": [
    "sha256": "9cdfbccffa9fa419932606fdc0bebef21ca27197662e2e09d9c909f193f4702f",
    "sha256" + debug: "4a8f50060329a2f2c39deebf92ec6abe35e5008a8dc29036fca24b14133b195f",
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
