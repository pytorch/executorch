// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241121"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "afff5bd4535392bd35fdd7c1372434eba61473e97beb5b582dabb4dc134b98af",
    "sha256" + debug: "2985825a09a218a795968de2eb207ae0c33e3266a406ee63ee0e64d1b250a5c9",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2909f460f701508942e02f8f911c63c49481338e8759bfbeb1c61474f0f82502",
    "sha256" + debug: "463380bcaa59a2b36c101feeae6fa44971af66bb1f132db6aaff017cf96e4ae6",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "cfe32afac787f7db3bc1d0572a77851168a4704850a5a4c1939f006b873311de",
    "sha256" + debug: "47f958af233710b0cec7c1394bbea110137f2f7613850cfdd656494aa1808a3a",
  ],
  "executorch": [
    "sha256": "7d209c9e01ded09e2c2f82b4db7002c6d5403a3bb2c38b54e14fa3ef5ae80393",
    "sha256" + debug: "d8d4c155efc174a36ee75b746b50b1ea1c74e3b696eb3b947da324386eab165f",
  ],
  "kernels_custom": [
    "sha256": "df5e822e6e0c85ff679225299234c5ce2885b7f0432fc7c543ac95f8c2798c06",
    "sha256" + debug: "50e748ead81cb70124476f528b12542db8429d36c4ed163a7f20f23c391dc6a0",
  ],
  "kernels_optimized": [
    "sha256": "2cf8e6cb4abbb966fdbd7a82fe742fe7111857dd10595fc754955306b9fc92b8",
    "sha256" + debug: "283d647300cf79e050c80a7a591351c7f5bd5a40eb8e6a9b25b313061e82998f",
  ],
  "kernels_portable": [
    "sha256": "efc33a0b425915490ec11ba44e8f4a349e03e17c121bc59a8c14d68dcc1af795",
    "sha256" + debug: "301ef2089ddb059034828d494ceac2466aed5ba4262ecf5f0b88af1b02b82b8c",
  ],
  "kernels_quantized": [
    "sha256": "f338c715df798d5334823698088a6e66c823efca17d8867b5e7be7955413c0d6",
    "sha256" + debug: "9bdf824f5896fafc2b922c2849772258eb9a6c4c511af7238786cbdd18640331",
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
