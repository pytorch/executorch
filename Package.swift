// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241205"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "1f4601a55128eb4fc5fa525dcb64245cb6b4fe4638bd5753ce5dc7b3b5a84672",
    "sha256" + debug: "bbfd4c918e815fa074f556d79b8adcc96822222062253714ab953784cba460bf",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "f929ab860cc600c5a2a60f3c20ab2b9eb89e4a2426c5584ca14d1e5c39699bc5",
    "sha256" + debug: "155f89332ee35679071e103e053efd37c2f94c4a88511fd737a77b87539e6431",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1def058ad3297ff10468a2a452c992beea7a546d2ff59bd7787de9ecdd5be67f",
    "sha256" + debug: "8ed598e1f8fe75bc41e2eecd671b2db4f2574bdcfd3018b82d32dedfd888d443",
  ],
  "executorch": [
    "sha256": "8d026d9a6e26c930f440886472f58cfe7866dad3b9229142d5535b8118829ea6",
    "sha256" + debug: "5aa5323134ec2b3e3c1acbdd3dfee5c4fe6ba67209b6f8254b033c5e58778235",
  ],
  "kernels_custom": [
    "sha256": "18ded9ac2a1a54f52945b8db94e2be1ee78c4f9f064f0a102e7b38d558e3998d",
    "sha256" + debug: "6aa8e74842d03dd41da04d49ebd61a5cd665fdce0798270aa9c295906bf01734",
  ],
  "kernels_optimized": [
    "sha256": "a5b302a9ca0f40030d0486f37a168891b3850c8f4f6555b9cd66d7cddb1f8b1f",
    "sha256" + debug: "a0f7db9c59428f6331588a5affdc034e4754ce6fdcb99e4cdd040a473c77eb03",
  ],
  "kernels_portable": [
    "sha256": "cf07596fc674239280f0400f059bb1b1ee2f4e3e402f373d0b1a94b4228ba971",
    "sha256" + debug: "0ea5cd83fc30bea5eef38768fbda453f3ce6f0601496e42f58146c03517a7ebc",
  ],
  "kernels_quantized": [
    "sha256": "edd86d7636f9724a5c0344e3c9c55e5333697cb0b839e6a2f2d9c8fb1a429ab0",
    "sha256" + debug: "c21173e17149e7f4a0aaa98779617270b1af825d3ea01367ec2c8894e5c5f219",
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
