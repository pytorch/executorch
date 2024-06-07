// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.2.0"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "coreml_backend": [
    "sha256": "1d81d3719812b6f26c518479dd999520266bebbf3c867c1b686c0ac43e467555",
    "sha256" + debug: "25fa8e4dd74d253d0f3fb66192f48c1e7384fb0d110003928e3b9aa25f22fe14",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "custom_backend": [
    "sha256": "ec1b94af1b74dadb5c78ea482366ef445dbde43a62cc2ee49986356f1983e049",
    "sha256" + debug: "8e5b399f6750ee97bd7373a6d78d179ff0076742ec3e718351803659a6219a92",
  ],
  "executorch": [
    "sha256": "8b33727b760bc96b0ec966a3a1ceed205f930a42a921c486da1caa91a6fff3c3",
    "sha256" + debug: "bc4220b1417ec1ff63092ff4191c00ce82b20b4da1d11a4067083b13c6f9dd90",
  ],
  "mps_backend": [
    "sha256": "1f06d36b52d4d24d821387f1e17b5f1f64d93e9ed2ed6a4907ce17436c61c603",
    "sha256" + debug: "1eb38e9cb45f320d45489b9e54c0f6468274b2575d45e5f921d1d88b0cc40df4",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "optimized_backend": [
    "sha256": "f1e0ad6905da488aa32056756dddc629b6ab565e18a05bb721267a5be68ae859",
    "sha256" + debug: "3ce0a8be0e9f6abfd32836b7a80c036efbe909b5870ec6fa32e9f134cfcc3afd",
  ],
  "portable_backend": [
    "sha256": "079d41baaa6c2decbfa2061c62cc038a7587f9caf740fe58e63cedb85c8ff998",
    "sha256" + debug: "492bb0a6c7941be713d8755aa4d968b019d8a7aa9e63cef90a8a32d21ff6cf76",
  ],
  "quantized_backend": [
    "sha256": "f670ed609cf798a664425cc7fb1c3dc56d40f5a5622980b7d9ff5a277de12bb6",
    "sha256" + debug: "ba99d1b31a2414a30131c53cb6bcf202addc263dea5785921490aa26769bc01c",
  ],
  "xnnpack_backend": [
    "sha256": "fbbe1cdb0dec8efd0d2e2b4acbd503230113404c7f3e99cebe36d196f845f662",
    "sha256" + debug: "d1b1b0e512c8cee2583ca7c9ae71fc3fb81dad11063c216866061cfc323b2ec2",
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
    .iOS(.v15),
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
