// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let url = "https://ossci-ios.s3.amazonaws.com/executorch"
let version = "0.1.0"
let coreml_sha256 = ""
let executorch_sha256 = ""
let mps_sha256 = ""
let portable_sha256 = ""
let xnnpack_sha256 = ""

struct Framework {
  let name: String
  let checksum: String
  var dependencies: [String] = []

  func target() -> Target {
    .binaryTarget(
      name: name,
      url: "\(url)/\(name)-\(version).zip",
      checksum: checksum,
      linkerSettings: deps.isEmpty ? nil : deps.map { .linkedFramework($0) }
    )
  }
}

let frameworks = [
  Framework(
    name: "coreml_backend",
    checksum: coreml_sha256,
    dependencies: [
      "Accelerate",
      "CoreML",
      "sqlite3"
    ]
  ),
  Framework(
    name: "executorch",
    checksum: executorch_sha256
  ),
  Framework(
    name: "mps_backend",
    checksum: mps_sha256,
    dependencies: [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph"
    ]
  ),
  Framework(
    name: "portable_backend",
    checksum: portable_sha256
  ),
  Framework(
    name: "xnnpack_backend",
    checksum: xnnpack_sha256
  )
]

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v15),
  ],
  products: frameworks.map { framework in
    .library(name: framework.name, targets: [framework.name])
  },
  targets: frameworks.map { $0.target() }
)
