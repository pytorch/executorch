// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250203"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "76ae4ece0c8a752d4b75b9bd00fc68f08d631b8fc9e3b536dee32017ac10f218",
    "sha256" + debug: "265a3290651524edf614543ffab97dbf8710aecc66648fb8c99b04786fa63e5b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "746cb3bdc367bef201c0b032ed15276b0504a5cf31a5e1a4558589ba9711429f",
    "sha256" + debug: "bd07ba2675f043d5a1e968e51eff01f384a623d0dcbe84ec30a1f2db2806898b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6f0621a3dfc2dbdbe345de76b780e7e82aa66251ba69a1e5834e65f2cb99b865",
    "sha256" + debug: "c6c3927b20ea39123a5b9a14c49d7b811f4810b2a51129e251b22911c2c1b7a7",
  ],
  "executorch": [
    "sha256": "11115cedf2dce3e2dc10b507933b3b5dd6723844bc900dd994c080a412514c15",
    "sha256" + debug: "aa92ca20568068840f544504fe434562e8c0f0937f6a732365cf14ef8b15ea43",
  ],
  "kernels_custom": [
    "sha256": "b6d3a38ba1ed46bb6b975aeadc4eb5c2141bb1cfe2bf6a86ad3a095bffa21574",
    "sha256" + debug: "9a874a268c5ce7737a27a47d6010c357e5c3d9b1ca9d218d2c4e1efd6a4e6e70",
  ],
  "kernels_optimized": [
    "sha256": "1a2cef109a4d1f7a876ee8fcb924219d5868acd4d6888e24dfbc8ba1292bcd89",
    "sha256" + debug: "c693ac652cfeeec90bb0fa6f8ec5f70ee9d11dd6b1e9f61889426d26958c7d02",
  ],
  "kernels_portable": [
    "sha256": "f08d2cc1a4d71c0eade60b51d98ba11ee128579b7c2f05a4560269b3579f3b66",
    "sha256" + debug: "eaf3be30abce5cdea09ccea92d057fae6f1d17f6d9acdc5b3cd8f832298f981b",
  ],
  "kernels_quantized": [
    "sha256": "20bf47d091c8721aa39a002320496ce86a7609c43ed9979f554b17e7834c6820",
    "sha256" + debug: "87b0f5d26220fb5b7a15c5d4099629b67ffb5b460af35312b48771d11db652a5",
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
