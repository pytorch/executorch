// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250115"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "bdf2cdc9e23480bb9d046cd7c8dd4a479a33d42ef7ece59e6b5e193bdc634189",
    "sha256" + debug: "ed48ab8d89f5f8027eed6dbfe3843e5abca00a89fe89604571a97831d4867c6c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a539c9bdc354fa34e6efe8516505ddfeaad1818e8fba1469de83584075e6a72e",
    "sha256" + debug: "80efbdb8d3c296aa2daa2b715fe72e3a2a1ad14c895ce5222cb4c63df4936cac",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1586660f33539da210d20b80373fd3180b6ba046477a38287e6771a99cf0a3e3",
    "sha256" + debug: "12c8116c2981c6938733d0dc9cfb4311297bb5a73a5426e53675f858d2f80468",
  ],
  "executorch": [
    "sha256": "814f2fd0accbeb6f859f733125e5d84c90a64e878107e13a4f0f4dede4623ae2",
    "sha256" + debug: "8d7c63a4543c4f0a5a967fb8c5a133f3a3231ca92cfe2d8cd9486f95538fa2c1",
  ],
  "kernels_custom": [
    "sha256": "0640d035816963f4a9f38717b22f6027ba5a09dcb20d17b1d3ac358968e5157a",
    "sha256" + debug: "e1463bc5728b515001cde2478b7297e1b19f55c4c8ce1f7486d2a11a2703ef92",
  ],
  "kernels_optimized": [
    "sha256": "773b00d23a76c3ec64a56da1f66b99efcfdfb8ba9193ad0c8b6ad46ab1b01d54",
    "sha256" + debug: "a0be0d7ff4e352465c1243025133e7652ecd5ff818841937f89973f3effcc563",
  ],
  "kernels_portable": [
    "sha256": "16f853e8ba9f7b862eff054917241632ec2f77a60e4d7c852be1eab2b3481718",
    "sha256" + debug: "72d42abffddbbf6d87d7c5b1f28f0f92c54487559acc88435595ce691f099354",
  ],
  "kernels_quantized": [
    "sha256": "df0041d6151fcb59b85297f3b1e287ff7497b8d304995c36cdb38a4c621ffe0e",
    "sha256" + debug: "81a69eb0cdade147796a5cdf591d355031d10ed2c68a92d7796a41b8f0782d00",
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
