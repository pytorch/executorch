// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250224"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "014b2d4d57642f1b2ef6f7d2163c6a7b2ea09988439d6e4a9774822941d62c7a",
    "sha256" + debug: "570f5c8e538becdaf9052b395d0f20c4dfd2b62bb3650de798f5469eb9a68e78",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5590207c21b7fed03c17c255866d767a529f1b801caefc0032c29df6af91840a",
    "sha256" + debug: "663022259f8d0b3098d47a2eb0e865ef4065a0ebbe9e9003ae16748f2f391f13",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d2bba198b549e004d67f2e8409e354a143fd7425b46c7efd6d9a1705abe45916",
    "sha256" + debug: "28dc6b6e16ed8be9c79ff49cdfb54bc487e1bc7f731e101fd9a55243e4b3533d",
  ],
  "executorch": [
    "sha256": "43799d3644686dcbe6b9f597f38ab8f2ee37780edb023655d68af30150240ea4",
    "sha256" + debug: "7c53057be1ffbf22c1f5ea15f4988c3790e7aa290714ce5fd8ba585a2fe2ce3e",
  ],
  "kernels_custom": [
    "sha256": "e9159a3765ddce939b51de23d3e877cdccfac5166a12bd897dad9beb3729927d",
    "sha256" + debug: "c9a6489ccf7f849c7fb109298cca3d0e706187f1f749479e56fcac054aac9e8d",
  ],
  "kernels_optimized": [
    "sha256": "b2776d3aed98fb34da2d8b08b558963285ef7a26f0a763751ab903de478e3f70",
    "sha256" + debug: "6b4735c67b3e35e5d28fe249f8e9f17bc7562267c501a68a1c04457865f2fa8f",
  ],
  "kernels_portable": [
    "sha256": "d8dc21fd14cea33871a0b972bf83c2cb8b2525a69e649e15356738dde820c10a",
    "sha256" + debug: "7ed4fa69f65893e143f4d111e525d0ff233dc9429c4b6f5a884aed4f40505f83",
  ],
  "kernels_quantized": [
    "sha256": "16d004d6812e2a49b2aacb4d9856bbfb97dc5eed292bc077bb7675b9a4791f19",
    "sha256" + debug: "8c7498e8c539f9182b3869b20bc3438e032b9cdbddbd1e9a33341f77fd8bf170",
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
