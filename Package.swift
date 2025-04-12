// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250412"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "afc430ac71e244b2bdc7a7c95a2017ad9d897a6ce9a6b552ed594b5785bf8781",
    "sha256" + debug: "7bfb73844508c15dae9a550c0d1fae04fdb1c7bc335f2fc1b6a1a7a9d2210f80",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7c822e4e2829cf42588f3e708ce3bfa70db593c1b2d991953491db93f8952364",
    "sha256" + debug: "73b167e5cf4ea96db4e88278b6a970a18d6fe2298d85b4e4f1cfc2573c70087a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "24b94725a92267cf66a125b2695fb89136aa6605d06ca10043db70bdb9a324fa",
    "sha256" + debug: "bc155bde5fef9381ed05d275ac758d6904e6586836a230467af8291f43be405f",
  ],
  "executorch": [
    "sha256": "632320d2428c7a40f22e7cbc9ee2210e4008d3b070487a49408852f626d037a4",
    "sha256" + debug: "9e7bb3b43cd9a0cf39b7032e3232e98b59752afd9a403d7e7eefe5af99756d63",
  ],
  "kernels_custom": [
    "sha256": "99dd22bab4cb30339e1a035d02411492fd1c255318c65f97c16c6c41d6adf363",
    "sha256" + debug: "b950d8ad426da16f68977492b7792faeb8daf8c2206efff597558d3a92fef3c6",
  ],
  "kernels_optimized": [
    "sha256": "52459ab53e85d26080d748f21cf025ec19e6fdb2536ea571a3deaa9f2caf3d72",
    "sha256" + debug: "f66bb138cd8fce3f2d30e9ba6367e1457ac8fa3cdf0ca879c6d834d8964fb2bd",
  ],
  "kernels_portable": [
    "sha256": "edff4f431f2899d804d13105497ddb247010886c24d1936739bba06123b723d6",
    "sha256" + debug: "31b0c1d995f1328a0cdba44067c116eec93b8a60dd1b029baac63311c25e28bb",
  ],
  "kernels_quantized": [
    "sha256": "142a13eac517298b4af5a7679868fed0610283a3fc75e4da99f4d2eafc28bc94",
    "sha256" + debug: "3609c6fb1331dc42247c1cbbaf6a75018c52d248be429805a95536ae10102c3e",
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
