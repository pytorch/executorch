// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250424"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "005abe42810b54c037b0a70571edc9cbd2301d63c6efc045b31f1aaa9886e5b8",
    "sha256" + debug: "245db6e55812e3ecce967e2b2787ec151e5bae4cfd214a9e42f4fbdfcdd58682",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e92e8f1843f65ec2d67f36a2a1e683398858ea41acbdc20091b3b9b4880bb953",
    "sha256" + debug: "f12ac55d02b8cfcd0111c47a8d87b1fd6388050c904955eb4e7d4156347efb08",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f10230fe5cd48731a6eea6e5b4a95979a16620f244f69e0478ccfa10c56ea7dd",
    "sha256" + debug: "9d65d1f21783acb76e4c0f8492e7ab482897ca8a47c04cda30da5c2475d9ad25",
  ],
  "executorch": [
    "sha256": "25ef0c8193be704698363d7dfea4ffb0f91aec6a1c3bdf61a3ac0d87ca77d593",
    "sha256" + debug: "a136ead18536ac93701f99fc8b8dd524400f8a57558dfc5be63c55f2e06497d9",
  ],
  "kernels_custom": [
    "sha256": "54ae1c5fe9790f6c2c4b2637220eaead3c9493a40f1078de7913239d0a82e923",
    "sha256" + debug: "9f1e419396a5d174a737e47a3ef6a8c7d0ff99b80bfb04e6f5f557f8b55e29b3",
  ],
  "kernels_optimized": [
    "sha256": "c335e12c38d2591d10e94842aaaba583931cd95e94d59b113002b250e730646f",
    "sha256" + debug: "95330c2a83413c673e2066e1a2af7a6aec6364b7199313b97d2aea6975152721",
  ],
  "kernels_portable": [
    "sha256": "2689d21c4bd7642005d80e84b5af73f5f203926dab48e7ab303d93ac145e9756",
    "sha256" + debug: "c6b8ff94431dbfbcb2bf69be87209f1c510626279b69f3d3bf002d113b1e5deb",
  ],
  "kernels_quantized": [
    "sha256": "80db6b957d1126f384374503b6b7ad52ddd63bd3cadc7962bfe824163ae1ef79",
    "sha256" + debug: "1287442ff6a4a4daf522440c5d9111a82754b69d33e606175d05e8219caf7f9a",
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
        path: ".Package.swift/\(key)",
        linkerSettings:
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
