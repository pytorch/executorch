// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250106"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "43f3207aa0180dff9fd8f0a705b31a4fffecd3e93e79ca3f4ee5088a8dfed9e5",
    "sha256" + debug: "234437ad301d01e834b2dab5fc94f5febe22936302a9875c8ffe8ed3398082e0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1077e2aace0b5f809605bde1586b7eb5d587675f87d83e9812f3858210fbe19c",
    "sha256" + debug: "2e0e792f64f4b344a3a3877c8beb52f353bde83823604b534064848a0f5e123e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7f3e2ccf9698256b2be05a7169d26e9d0be2d309d5d53464a75fbb9e688a438c",
    "sha256" + debug: "082383da269e39cfaccaa00aea58935ca9fc38de6d78193f93a4800ad284bf92",
  ],
  "executorch": [
    "sha256": "db03d87dc30c031957aea3ba8d450fb9f0d462a1f48188ae3859c8e79d39e34c",
    "sha256" + debug: "8723503b9d5c788f065f4a03cadd58e5fdc229b19b900632afdc00d05099dfaf",
  ],
  "kernels_custom": [
    "sha256": "977c67ff7b130a26656f037269cf53610a2c1f6c2d323740ce5eb6b58e13ef35",
    "sha256" + debug: "f3abaca8a310b4a8e8fd66b8c1faefd8e213f795ccfee9ac0b848c30d8c6de0e",
  ],
  "kernels_optimized": [
    "sha256": "8f2ba6346b6737747e337bb5d5903b3b1d22d5ee5e37ab88816f3645c32171be",
    "sha256" + debug: "f0927a2cf2a4da0c5ae06fa35baff4747c152a40acd1f732888ca47d71dd2b78",
  ],
  "kernels_portable": [
    "sha256": "ca935451a4626d64ec85ae16ec7fa6fc4c0d5e964cd8ea8c02f244f544ac53f5",
    "sha256" + debug: "0592af48e72e0bc436dece532f4e9708e508b97612e99233e5f33ab0594eda71",
  ],
  "kernels_quantized": [
    "sha256": "629346a226465f79460733922ebb71fbab3dc53a7b16c8927a7f4a64b9bd223f",
    "sha256" + debug: "bbf5130819f5128e67bd248d51be1508e1ef27072ebea852be2e4dcb793b7be0",
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
