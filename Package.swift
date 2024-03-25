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
let coreml_sha256 = "1f0432ee5782eab259ce9e3007415006be152c27ce530c1ef8a79bca940c6a32"
let executorch_sha256 = "bf1ea573338bd30e90433bfb93db5d713aa7ecc351579ee7432f57fec329441b"
let mps_sha256 = "df851be40444509f622ed155b787073aaba7643669eda36b6b3eb902e29b43a6"
let portable_sha256 = "f183c5934491047b04130304466357959ae65e99b0b482a8a3372676ef817777"
let quantized_sha256 = "bf79625d32161ac1892e172ec69e83a94b60ea400c8fa32bf4c4e85c7d338e8a"
let xnnpack_sha256 = "85ab60d6a29fcebaf7aac1eb775ef96c1d87d1fa104458d0efdc681b7ea6b558"

struct Framework {
  let name: String
  let checksum: String
  var frameworks: [String] = []
  var libraries: [String] = []

  func target() -> Target {
    .binaryTarget(
      name: name,
      url: "\(url)/\(name)-\(version).zip",
      checksum: checksum
    )
  }

  func dependencies() -> Target {
    .target(
      name: "\(name)_dependencies",
      dependencies: [.target(name: name)],
      path: ".swift/\(name)",
      linkerSettings:
          frameworks.map { .linkedFramework($0) } +
          libraries.map { .linkedLibrary($0) }
    )
  }
}

let frameworks = [
  Framework(
    name: "coreml_backend",
    checksum: coreml_sha256,
    frameworks: [
      "Accelerate",
      "CoreML",
    ],
    libraries: [
      "sqlite3",
    ]
  ),
  Framework(
    name: "executorch",
    checksum: executorch_sha256
  ),
  Framework(
    name: "mps_backend",
    checksum: mps_sha256,
    frameworks: [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ]
  ),
  Framework(
    name: "portable_backend",
    checksum: portable_sha256
  ),
  Framework(
    name: "quantized_backend",
    checksum: quantized_sha256
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
  products: frameworks.map { .library(name: $0.name, targets: ["\($0.name)_dependencies"]) },
  targets: frameworks.flatMap { [$0.target(), $0.dependencies()] }
)
