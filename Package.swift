// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250220"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "3c3bb8063b1edb34d7a42cc00706d357ab1c3d16d80f10846c50182bfa17f369",
    "sha256" + debug: "ad5a7c3ecaf2622395eb09f3d2fba97801b661b8be0512873398332a32864430",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "469228ae6eb12933077538a1ca11b778ef3a91f81070865abc7f0b280ed9834a",
    "sha256" + debug: "b89237859f2df03d7b23f45b1e577fb2533b9c77c62ecd7e99a890a462e00989",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4cbaf53c737da74efd842dcf77696b53055eccd0c2a92e1dab6fa4137ab6d2b9",
    "sha256" + debug: "bebf8269ce80359496d2c20e924d595391a86a6e8694a98ed8eade4616e45326",
  ],
  "executorch": [
    "sha256": "b730f1f63966419dec891ebf88aa49f2bd908cdf25798e0838147a5f6153442b",
    "sha256" + debug: "8b14b82d19ffefbd943c5742e8a59016225f52bcd444fa7e979c304dae7cfa37",
  ],
  "kernels_custom": [
    "sha256": "be359247bdac2ae7f2221d4e23cc08110d37e78b9dee1351aa9aecd361488a73",
    "sha256" + debug: "42623ca54c6adacf142279c9e9b9e343c7b7726104db00680105b83711405ac4",
  ],
  "kernels_optimized": [
    "sha256": "2b03a11007538aba769d5900fb53d02f33cd799cbb469a652c058b52f1741954",
    "sha256" + debug: "7231565a82afeeb9ea85f5348bfe7d8a7e52ce5a6bb6ad2e9aabec218f8125e6",
  ],
  "kernels_portable": [
    "sha256": "e67e00303714d5785e598d6010865d860774cf7223be90949cbd670b738503a3",
    "sha256" + debug: "2cd1c9234a9e3ef072e5c5b9bceac609677f4b3da5cbf9591522920a726dcbea",
  ],
  "kernels_quantized": [
    "sha256": "734e87499edd57e51df23816fe1c179b852729803adbb91a76ab17c3d0775a89",
    "sha256" + debug: "a321b7775fe9a7fea244a02a32ba653f74f3eef34dd179398a0c95e0164d9866",
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
