// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241223"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "c75930d559f8ad17ef053501ff0144fc9af89fe9591973a4095c69db154ba718",
    "sha256" + debug: "ab3e71381205b468c903c98c13e43a5f9b9c7ef8b18ad81163ccc0abd87d2d1c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "f6e647d88ee08ea876583adea79549c031e3294410740c3a6d371dd325a0be62",
    "sha256" + debug: "22257cf900e1ba5923d2c8cf2f9d072fc6812aca69703efba8693239b627465d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7bb10b35f990e0806ba857c76913e1b3f7b94c80d00f1135755549c90949c747",
    "sha256" + debug: "273c97fc7cb15fc7815ab1970d185855e757197956c2013d9d39d000f08014da",
  ],
  "executorch": [
    "sha256": "ca1852f19cd6c5029ceca6edf1213b7c1e57477fd8220231d0d24db803f4faa0",
    "sha256" + debug: "1a9027f0113145dc6fda180ac8f744a852fc257d126d9e86ba20c5f03806c1c1",
  ],
  "kernels_custom": [
    "sha256": "562f680f9c0282e478555d5e8b9fc93507a4929907c0ffd2c3fe33d3482b445b",
    "sha256" + debug: "5e7b5892cc748c7966569d4f821cc497d48683a9faefc42b40cdcf820cbe7c03",
  ],
  "kernels_optimized": [
    "sha256": "f92b07107d63f462eab17d2a80295d818954f5c7d7e384a34cb5fdd94f370344",
    "sha256" + debug: "750bdf649497398eb04c2063b5445fb7787bf251c60d64cfd1ddac35e0082a6b",
  ],
  "kernels_portable": [
    "sha256": "7ce57f1b1e9a8ed2bb21769390f4d7c4dccf8e2ba1f177d43e052aca2df49e57",
    "sha256" + debug: "c02a70505f46fd70441037885e229314fcb5560caa5a9d7504c55301c50ec2c4",
  ],
  "kernels_quantized": [
    "sha256": "a14300247f62c5dbe3c9bd49f312977b6625ede674002c3ac07c711130cbb273",
    "sha256" + debug: "cfaa617a5c5963cc4798058553a3ed81c6aaf3d8f25b7b00716e43b435ea00be",
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
