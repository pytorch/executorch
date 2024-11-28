// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241128"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "fcaa3ce4cdc3e423535796ba44d366be6675573ef6cd061dfa214cafa25f9ad3",
    "sha256" + debug: "a0c963d67f2cb223dbb92b57a070ac398244bfcc131c0be64a8c668d56080486",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8ce83e8f865b13b90b0b6a4257116c768264e78cec6420a06273bd4072d37c04",
    "sha256" + debug: "54bdf8128a7c1f301ad5ce100b3c76a94d4dfa0ed34f93f5e112e1d1c41d2aae",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "631f7a5c7cf0edf8442a8605d883cf02366d24039cb9e0542f6b5985fbff7463",
    "sha256" + debug: "0384b51fe569f11cc38436aed4b2033ba2f4eb5aca49bd8e04a9be828cb6a905",
  ],
  "executorch": [
    "sha256": "ae37c5afc13f228eb21fd887f0206876b4498fdc695a4e324291b72eb337f0ad",
    "sha256" + debug: "2927d2d0eb8833d292ca351c0e517e3d76fa09eb7cbf875a96e00a8b5b05cf20",
  ],
  "kernels_custom": [
    "sha256": "126f6d588b42bc9ccf7e948d126e755d55accbfe4fd42739fb35d3cd084422fb",
    "sha256" + debug: "e2405002973aff656507a4015c482d7ba23a99d1c4c723023606ec96f4fb5896",
  ],
  "kernels_optimized": [
    "sha256": "aae414d1d86ceef5318c67be4cdb732de8cb7d60341dc87890ec515f2296e09e",
    "sha256" + debug: "994cbb608a293670e1592b0ba62ffa04e9eaa83ae4ad934e59a2e05a3b631c69",
  ],
  "kernels_portable": [
    "sha256": "742f1fc235ad65e6f6155b881434d31c652bea7c5c8d1fadb5491b51799202d2",
    "sha256" + debug: "8b58fba0d65e7e7d7a805e7d8a22ec791c04058c07188ef4c32a4e4ff1b2581f",
  ],
  "kernels_quantized": [
    "sha256": "032edc2debad9da612d33ed5975d9a38c66e2fc869865c5fb3b3f1059c748660",
    "sha256" + debug: "97ce18c358ac36dae4820026b0c5fd3209763783120d26bfe6f2c42621734412",
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
