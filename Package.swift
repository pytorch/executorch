// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250225"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "165fde17135557d0313208dff4d1adf9f9de2d938317371e4b136317f7de82d6",
    "sha256" + debug: "95fed0c0d74e1298fe57279bd1ac5852c1907df126994427afc33b661a2740b8",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "21e1275c82eeb2beac47183b309170c4201efb8d5478dff4c623c5bfacaf6932",
    "sha256" + debug: "ee084c0582d0dc31dbb2d82988c6b0ad35945314d5a0991ecb378051e1fca516",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1b75609cb00c2bbed2ec3fae96ed0f417171d41e17c31ffae3db830c82449736",
    "sha256" + debug: "da22d3d7258c4cdfe0d3b8febbbe60b8b37de46af5b336ffe8f68966673d609e",
  ],
  "executorch": [
    "sha256": "367e6fe81d53195722442b36902397904d580742d7c2117df4793e1d9cf2adfa",
    "sha256" + debug: "59ca50973b105f085ea727f06da5d116f298dec85059208dd03d789b2098ddbd",
  ],
  "kernels_custom": [
    "sha256": "f74b1d3f54eee8f10425f9d9fcfc0d5ce802a9ad0bd803a83886d1a2c0c59dfe",
    "sha256" + debug: "fbe9ed544aa6a1638f5dfc345d265288a2f940859a9dfe855968aaf1a9e343ce",
  ],
  "kernels_optimized": [
    "sha256": "080c98b5e8ebf106f597ad62dc3f461cc33ff344aa53247b1173484ce7ac864a",
    "sha256" + debug: "bfe47470e19e5a023dbc9d9df3a7f72cff6f56b6c737c56e96bb2c580b4f7796",
  ],
  "kernels_portable": [
    "sha256": "db996f4216456450e9db8899f53569d6ef7d63a2f3e4d0bd744af1af43d14508",
    "sha256" + debug: "5a7c6e6cbdb7f5951dcaac2c699464f424d498dd4cb4803d69f0545a4b89a461",
  ],
  "kernels_quantized": [
    "sha256": "ba9f4669d2e066b50700fd2f7b17fdd22d3df786547628030edab9c112046493",
    "sha256" + debug: "38cdb5450bb9270a9275ac9b255a2e5616332b3f5d6e5735df47e8eb8f20db19",
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
