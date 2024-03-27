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
let coreml_sha256 = "a08d3a06f65c6c124214b27de77057832452206625cde36261b4b6a346314802"
let executorch_sha256 = "ee0c1b870036834f7ac0dbf99fa396990243a96e0939c7d4f0ea341b794dcc38"
let mps_sha256 = "020fedd9f7670422c132da42ddf3b9307c67f12f85c6928109f1d4885c67b1ca"
let optimized_sha256 = "e5f3d9814758d79da7547c1936e7a665e305a82e4d6f340e25e41b6b924e45d1"
let portable_sha256 = "968a8aa09794b69d60c9cfb6c9cfc37c8842a51fd0cafa14f7b7daa4d8e80eea"
let quantized_sha256 = "e46e4252f5d0f134bf2edbf559ad07c92c49288dfcab21fa7406e1424051de1f"
let xnnpack_sha256 = "016d4b3f947c267d9ffd4884198730a0f5a5a606d3376addd96e45aaa7a366cc"

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
    name: "optimized_backend",
    checksum: optimized_sha256
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
