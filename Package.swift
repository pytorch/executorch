// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250429"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "cbb7728b0ea752b8b029eb5104d1177399e1e9c16c0254dbb5ced6a7210f9e8b",
    "sha256" + debug: "016b5e743fa035d16dc9795c0c3aae0b4872143741c851f92be2dc536acb2e91",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b90ddbd9de9e789ca307cd1531b20045e1d7da6fbc70eb914a27941dd0d564f2",
    "sha256" + debug: "5cfd3a8d19270aee1d774af47fdfd9d1dfda25dc37eb366679a9cf36dd7c988d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8008d5afc5b03279da36ff5e010257cfbeb08a358821e3ce4034a08ac0ed365f",
    "sha256" + debug: "6fa265bfd6bd4cbed2157375093cf6636f328b6662983363724f663d242c772a",
  ],
  "executorch": [
    "sha256": "0a9739db250d8bbfd06bae40d5404a037327bee52cb277f59e13bb92611750d7",
    "sha256" + debug: "615d9e550e94a0008ae97edf4eb169ea41c29ec5dbdb9bbe8121f87da6505f7e",
  ],
  "kernels_custom": [
    "sha256": "835d9dca0c42431247248143d40e0522b03ac287f898fd01b25f7136aeb070de",
    "sha256" + debug: "40680c95872e9ac13f67639c391c7a9e6eceefea53a5fec9b847febc893eafe8",
  ],
  "kernels_optimized": [
    "sha256": "3c7a814227da2e69d2138e283ebdd616f79ed2d3e9302d0183f6185deb005854",
    "sha256" + debug: "3af158d20861716714011d1a73a0adca93bdc43abd66131e0da9867ce80d2885",
  ],
  "kernels_portable": [
    "sha256": "f3eec5fb75a469b51cc9031cdb383d5f444d1d1d09339f1b5bc22178d286b51d",
    "sha256" + debug: "c598628155709e4f1a597eaaeb9da678c121420c1e90b69ead6b567670ab66d0",
  ],
  "kernels_quantized": [
    "sha256": "e0b6deb3c6dd5168543969fd31ccf01a13b0d806c727f6a6102c1ee0f4caafe2",
    "sha256" + debug: "bfb27d5bdabdf6e69ba77586f095bafe70e83f5d048562571d64f6e109125a48",
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
