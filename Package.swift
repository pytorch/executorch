// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250201"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "2a6f37d623d1fd4a25514bedc1b78f5dab047afab3367b89c869562646db1280",
    "sha256" + debug: "549cc40181ee56c2b2e976abc34e41ef006c9eeba18be501ccb0a6b4abfcf779",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d37d8eac218bb1005f7f82c2b404e44174f21cd37618234319e3d617fc6f4e0c",
    "sha256" + debug: "30fc4560530cc62252912f1f935470f758c4bc176ecc76c8343b2b2176a3a87a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "45ec98892ec55f8b6e5a1c1ddcaed8f723340953bc1243a723c1a2f22d6d989b",
    "sha256" + debug: "d5c9ec47b64979f6bbeeaa909ecb69e0d11ba59ea62f3c1e0775e465cd94b940",
  ],
  "executorch": [
    "sha256": "6675dd7ec7763c287998f10dbd4f98d9365dd163fb7e21e44e80dee8267d1cd1",
    "sha256" + debug: "94ffc32873fe594ddb53bd7d699048b9891de667faf48d08648e4217d0888bf1",
  ],
  "kernels_custom": [
    "sha256": "6f859509271ba8ca6e1a2b918da90c6b0e2850fc7fb50fb04dd5483e8b184ee3",
    "sha256" + debug: "1ca02e81ec47ce926dc764ffdd0966cf6a8aa146c9f2088ec81864e0918396aa",
  ],
  "kernels_optimized": [
    "sha256": "c59b3ea246709916416d7f6d413fb241bee004cafac5b7e90151a0ed2a969ea1",
    "sha256" + debug: "bc10c930a53d600869e605e22d59b9ec618e3bd4cf6f5f60592256e49daff498",
  ],
  "kernels_portable": [
    "sha256": "1f19a1e161479af88b81954710b1dc334c8fef502062db671c6c0674826af950",
    "sha256" + debug: "551a312f4d7da4faff4871ae9bdc76497c944dceb176dba4a801e63743e739a0",
  ],
  "kernels_quantized": [
    "sha256": "01ba28bbb321586e239fe377d77ae5f775860484398faddf564c40861fe38154",
    "sha256" + debug: "fe4fa0ae56faaec39d45518d364b992b932acd16998b98a8e8de30bcf2cddc0c",
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
