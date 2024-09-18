// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "dec20ff247236627079b76788e3218bb52af93acd22774cf11c5734c19110779",
    "sha256" + debug: "e6bc819b7b217f840c90100a5f4150967f7a2959cdcc56f18089a10f674281a4",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4002ec7af976ddcf70f209c0f83cdec074aa2022a02fd4a6bdbc0280fde5d67f",
    "sha256" + debug: "4f2f116abc1db7fc9eaa809a50c0949105de559aaed73009f179a90b90deebf9",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "851a5151027b7658140163edc6b81694fd9ba4a57c7c41951e84a64960e0f199",
    "sha256" + debug: "0885d8412609aa6de75fd77a51d71a1f16cbf31459937fc6e353307966bf1cac",
  ],
  "executorch": [
    "sha256": "1c5811504e799abd88848da6449f2a03f8e4db6dba15027e5cd9c77cac276d36",
    "sha256" + debug: "ddc4a0aa031a0487c209a3d28b036d9879b029ee63a7b3a73469b25d75a6c1e4",
  ],
  "kernels_custom": [
    "sha256": "9f01a6314bff6a1739446c2d7b6a5a9cebcdbbbfc13294c8e173dcfd56312be2",
    "sha256" + debug: "fd7ebe492dcb36375d40b19a1fd9b7bf943936fc6934b05539c407ed9bfc4c69",
  ],
  "kernels_optimized": [
    "sha256": "f983946e68ed9422071eb7571b252c2992a024c02a6b9b3fac9a074d386b9491",
    "sha256" + debug: "b5f6f621b89a2868c7586e7e430fa6883a517c65b67ac46d722622066f5feb89",
  ],
  "kernels_portable": [
    "sha256": "c5c441bc8f78f549fffbcd9b0a733bd4855d50e20cbf3bc11ac5a09057cb7363",
    "sha256" + debug: "70046f787487a356046cde5edf96d5e08fbb667ce2e497cc575fb9e6e1d8597b",
  ],
  "kernels_quantized": [
    "sha256": "be479baf8b91ded3230d7538e86902d7ae233702e73fafeedc0cd2b76fcab7d1",
    "sha256" + debug: "701d629774c8565a58510452a0ece98921d96fba17b72d308ccc4b08d6abc503",
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
