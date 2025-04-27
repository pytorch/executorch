// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250427"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "dc2cb5dba2f8331e4415b7eec911debe1f5c1c74db80f040bcddace3f3680c4b",
    "sha256" + debug: "2495f658a5374b126cd4e672ab3d7f5c324678d4343b104ce81a0e925cae9af4",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9952fd4a1598a0b16ba9c006e20cf5becc24a35bc8c4e56b153af0515554a476",
    "sha256" + debug: "044c9b301e78ec3e88054fe8aac374a1264d8254d069be351c7166032c93eaf6",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e40aefd415e62411d6e2fc173b7e90f41f9b2e2b209a124c7a9bac519896cecc",
    "sha256" + debug: "1bd8d1fc9a9b7e3c61efa9c007dedb5caaadb5fc4a75154231d4a4a0415132de",
  ],
  "executorch": [
    "sha256": "6906561162770a3232ad4dacb494ed13b40fd0a87e90c5bff438697f594ada6e",
    "sha256" + debug: "b9784926fcedeb7a206fe12ee35bf960918a966b32a805542d7fc21993044c91",
  ],
  "kernels_custom": [
    "sha256": "52db081c06aff2a750a76de91b1bc6944fce594afa94b7349c3b0ffd8e6ab79b",
    "sha256" + debug: "e1f67f0bb3d0e64635c25ddc30979912dcf7ca9a774fd54de7b80dea4ac3206a",
  ],
  "kernels_optimized": [
    "sha256": "5eac7638619e3f954167a1a5e14e05c25e3583fdc1131be6ea9e9e06df44bc18",
    "sha256" + debug: "79b546306a0b7c69608b76cb0f884035f7087010dd2c6048d16e86bb0804357d",
  ],
  "kernels_portable": [
    "sha256": "5f39f51537ea4e25573b983c3ddcc103547fff166d20fbbae7323d16357ec4aa",
    "sha256" + debug: "2623894b91268094ecaa42e2a7dad31d5afb921f94117bfdec91a06dfe0d59fb",
  ],
  "kernels_quantized": [
    "sha256": "a04a05582076713e173e932917428e796c737f173601efb9373978b300d67775",
    "sha256" + debug: "ae3886de32c50a6b75d75099bafb2b74b891345272fdcb5585210e1f456947a8",
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
