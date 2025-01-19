// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250119"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "7812143067d12b41d7e8f2edc9516055c51e4791ce12e78b1f3f3236a5502589",
    "sha256" + debug: "24fd7aab3e6382b05847c45728b8b6eedf2830faf7780df3175bd8830425e432",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e034d89c3a3e6e8a9fa0ed061b438d6f174b8518869c645d17c63d76234cf9b3",
    "sha256" + debug: "bf43dde101ea04d27739ebd8f0c5beeb42f634f99620f3fe0eb000f14ea4edca",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "9fcb227f22d85ad26b5df02a3b5751eadcdd45ca12a458c1f04fb35feabb308c",
    "sha256" + debug: "6dfa2c3b4f0599b50676083511ce0680ac0c981d51db9d19d417435fcb5e25b4",
  ],
  "executorch": [
    "sha256": "85245af83047863fe99e7dfefa9771202f3491183b34ffd52b92ed922dda20bf",
    "sha256" + debug: "31808b8301a2222cd51c1f737dd81925c072d2eb0f64ea7e0f41dd1806408485",
  ],
  "kernels_custom": [
    "sha256": "11efae92f434728ce9a64296f46551e1789715ff8af27ee4221af37fb23d88db",
    "sha256" + debug: "2c283b8bd861b7cd02c3e90e11e07652329cf793646308b2f9e507c2e4cad186",
  ],
  "kernels_optimized": [
    "sha256": "a12da550fea71f6656da6213c30fc01b21fcbbda5496fa1353e31f51df31d6c8",
    "sha256" + debug: "2a7291926a7533e540ad7686593ae09348310a020285a3df369bf3bec11f7131",
  ],
  "kernels_portable": [
    "sha256": "96098b68ed6ef388fbc956542b4d4cbb978cc48cf46f9d81e6cd81111f3be1e7",
    "sha256" + debug: "24fd1ca3965b2e8e18524b2aa53792ba34f2b7f5ec2f75f7c6b134c7769f6d5d",
  ],
  "kernels_quantized": [
    "sha256": "931d293687bd8c1e0430d8fe687a55e0f95507cb6a5cbda0fdaf12b88b38b14c",
    "sha256" + debug: "4d7b03a2e54b39b52efced221c5430ec0186eba3d2437d5dea6755d81afe5c62",
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
