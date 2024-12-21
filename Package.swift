// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241221"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "93c40a745b2b6f099bb016a49bcdff3bedf2cdd22fbf0aca00995635e438c300",
    "sha256" + debug: "aa526f61c3c5c764a4528f8ce0b2d28be6466a4311ba0c20bce1c39c24b0144a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "57c10f636d7c9909ea62622c02d367f54d7a3e4a7680198d67a347cbfa29564c",
    "sha256" + debug: "055ebfcee59cfcc93de31a632af5dce7850a1e9ceee4a4ad028aa7025054d63a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7b29104a4020709d77f79b29dff38d501a7f229a7da8439a9e2fbcc2be9c0435",
    "sha256" + debug: "e3616837eaeef31b25e9ceffd4ef1b53a8b27b8d577333e6cde519794cbabf30",
  ],
  "executorch": [
    "sha256": "d8dc0c8e99d869e39cf4ef9bde6beb0b2fbf461d7c033014ea2e4d62443eabef",
    "sha256" + debug: "dc24454045df9f2039501dc25a960bb9590c969039aed77fa67d821f64673355",
  ],
  "kernels_custom": [
    "sha256": "3c234490f2d8d1d0d0381fc46a717c89f50416b26ef31474084be58a94458d35",
    "sha256" + debug: "e8134e6754b1ba672a1e91c4bf604d6de8a361bd0b4aa305415464e255a3cd64",
  ],
  "kernels_optimized": [
    "sha256": "b570a1fd06d50424ea7a6c2db9b740125538f6a6571d943680976de577389055",
    "sha256" + debug: "9e52e5c2dbb2c9dd358625f68fabfa29f9f1271b0ca7a4dbc8bc0964801a8426",
  ],
  "kernels_portable": [
    "sha256": "2ccfa2538aecd2e55425b319a3c9de534f6d448783d9fb88e6e38596b088de5e",
    "sha256" + debug: "1db3f4b43d1650005dd811d5e8efc702d924a668fccb4e526e505ef3c359b05c",
  ],
  "kernels_quantized": [
    "sha256": "f2c2ab456057c0b893d1ed023b634e0e728806bc55f79e83ac2a64e485d5b726",
    "sha256" + debug: "70adc15734d9c01779232a433502a6aa442f2acfde04c30bf206f9eabc057c91",
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
