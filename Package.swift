// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241224"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "21a712a067d0964f5846e858f456521372908bf280ad3c250fc94248ca997bde",
    "sha256" + debug: "c534936d86312dd91a9f5afb99b97845df6418ee3af1f21370b6d3b1fcc6ad98",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4b6d4401e47e9828bd4e646c28365d54d72d54e1034adc2bea2b514b725d5962",
    "sha256" + debug: "3c0a7a9ecba61cbdd4f419df6675717cc093dde43f2cbc0cd9dec55120cdbf96",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ed282bec9b33316f8df6ae23a84e9fae598a53a8427ef65ea290042d81175a60",
    "sha256" + debug: "c64e8668a3ef6353f5ab2295ab79186f14dc73b17633ce4ec5e250e4c99778db",
  ],
  "executorch": [
    "sha256": "60fc94e0a7359711db110f58312b3acfc69509cb06ff09e5d0ed14d7c0cc4a2d",
    "sha256" + debug: "51aaca05b5baf1995b337eedb27892a7af31b72b6fc832d8e2388724573f8651",
  ],
  "kernels_custom": [
    "sha256": "0acd3fa252761ad6b76584887627ca7c0270b906c531bd29b5927a8b6adf5323",
    "sha256" + debug: "0e41c733dc2104a88b6d0b75f148f6bac29006d2ab63c8e399e381e72ccbd3cd",
  ],
  "kernels_optimized": [
    "sha256": "73c6fa1495ebb5572347f395ff4cfb21a31fa49280b20e496582851a709202c6",
    "sha256" + debug: "22cba98809090184002c1c54d0719596a9c726530ee1fe05e464ba3a93573d09",
  ],
  "kernels_portable": [
    "sha256": "782e0338627a4e881a4419878edddd04962854dcf0ef075c4afe24aa077793be",
    "sha256" + debug: "b6589e90b1ddd5a2f90e202bb9d9f8f3ce344f37e14962020385618c2dcc2898",
  ],
  "kernels_quantized": [
    "sha256": "a5b4220132dbdb3ed9709ce3723f4d3c111c5be1f90b5a8044cc4db875563f95",
    "sha256" + debug: "83693fb8d50222859a97ddcd90acfdbe6fa23c420b2bb77629e642de29362833",
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
