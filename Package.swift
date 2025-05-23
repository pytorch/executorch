// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250523"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "75a8b2d6368cdd946da4e4c9a628140f85a4828edf95c39aab0f820b0b832d44",
    "sha256" + debug: "a972bad8e35b73ef7a9b9247f6e0718d8d2b6aab5f97cce207517f58cdf9f17c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3e6021a7845e89a711589fb08669dcd373691a4d69335f33d872b78b1cca72fd",
    "sha256" + debug: "c158c5a93aaf93a769d8af74727f46e6dc775431504d18f93e5097d9415d0036",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "92cd4b84ec7684d69c8cda1ad13cd5447f78620a51f5a227871b7cecd80b35b3",
    "sha256" + debug: "059106ffb9afc8722599d95efe8c969ef3489e41be00a2498aafceed6175f34c",
  ],
  "executorch": [
    "sha256": "59491b16ac52144cb7d893a5866a2cbb8408e561c65dfb55cf6d8ab323cf2fe8",
    "sha256" + debug: "0c4e14c4c9eda6474ef18827ec29dd405d1a518cd68a884c59768f1b02182665",
  ],
  "kernels_custom": [
    "sha256": "dd77f55d344b31648dc1b046ef54e0f013c34fe23466f846d1d71e2c87dc689f",
    "sha256" + debug: "39fc4f002b0141e385793092340ec96d9cd30ea9e3bec1a1672022b9f23a7bd4",
  ],
  "kernels_optimized": [
    "sha256": "5172d1be40895aa9e51e4d85bb16cd7a60c6ecb5af4571306564e03b91916573",
    "sha256" + debug: "5c705af518ed232bd48cacad014407723709d317a5084d2671c88ce768e0f0db",
  ],
  "kernels_portable": [
    "sha256": "20827416a55b057f4e9b20fe01e8f3e612d4d4fb261fd15e0d1c5d6d0f960bd6",
    "sha256" + debug: "127ca70872e9b39373734296691132296839b056ed1da74914f942bc3edd645c",
  ],
  "kernels_quantized": [
    "sha256": "e754da2f10fa9d1c5ac0f81012f67aeef17b0e78256986adc930976f2d4ec919",
    "sha256" + debug: "b5dadc10e0984d1f218374901a78b713d6b1c216177f254f5a7ead8b6aff09bd",
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
        linkerSettings: [
          .linkedLibrary("c++")
        ] +
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
