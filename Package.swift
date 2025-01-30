// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250130"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "3f45664b660c8b3a81d8f63a478169dfdabecc361239af4d2869ab0724642959",
    "sha256" + debug: "2afcffa51509cfff0a09956011a55fbf0ba39ee17d5f99cc35bf5912905adfdd",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4934343dfff50153508540acffb637e3618aa4ab63b439743e24f06f76764c37",
    "sha256" + debug: "cda498aa252cb9a1208d3cd00148382ac36496816405f812c45e55f04d9ad238",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8ebdff71cf34c488c988187165747670214e646eea45f97159c823443b0961ba",
    "sha256" + debug: "2a41c6bca0a463643b15c4b73fb01e48e664abd0c21012e83aef46a7cc1f727f",
  ],
  "executorch": [
    "sha256": "4597b56790576bb0ea0b096ec69ea4460dc65fec6bca29cca61be014e5fd735f",
    "sha256" + debug: "c675ea10441f2dd66525700a4167a1b9c499a0ab83712c4c05dda10a0ad22409",
  ],
  "kernels_custom": [
    "sha256": "0eacbfe0b5bce441c7d0c20b5fb6d9be3ef0d1dd91cc55880e351334dfb2b1a0",
    "sha256" + debug: "6dfbc4c42b129c7c56f4a291ac9fda1830813bc6f675028ddf073b741dcfb2d1",
  ],
  "kernels_optimized": [
    "sha256": "7601b85700ee576c38b11cb2a566e0037a16fc72a2d10ff4f8ea932dc0453a0b",
    "sha256" + debug: "e3cbf1e2e519770864b003fd16f322255db557503e8a1d05d7bc6356c28458c7",
  ],
  "kernels_portable": [
    "sha256": "aa4598e1a4e7fe594bc30bc846c582cbf390f73e36aa495ca5e94b6119570aa7",
    "sha256" + debug: "a20f87c93c40a95dbce3852cb4559c764bb6d81b6140db9d52fbe2c3f62ec899",
  ],
  "kernels_quantized": [
    "sha256": "5082d7437181b90468a6219065360b8944b5359a9f6b0c7b55aabc98c93854d8",
    "sha256" + debug: "7c931dbb7918e9d42858c7e8666378ea9bed2e161486cc94922580faa7760da5",
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
