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
let coreml_sha256 = "78d853d87be478696e56e658aa4ff17d47ae185a9a6a36316c821fa8b2d3aacd"
let custom_sha256 = "f059f6716298403dff89a952a70e323c54911be140d05f2467bd5cc61aaefae3"
let executorch_sha256 = "ba9a0c2b061afaedbc3c5454040a598b1371170bd9d9a30b7163c20e23339841"
let mps_sha256 = "39542a8671cca1aa627102aa47785d0f6e2dfe9a40e2c22288a755057b00fbfa"
let optimized_sha256 = "1d84fa16197bb6f0dec01aaa29d2a140c0e14d8e5e92630a7b4dd6f48012506d"
let portable_sha256 = "4993904f89ecb4476677ff3c072ed1a314a608170f10d364cfd23947851ccbf3"
let quantized_sha256 = "8d35ee0e7ca77c19782eaea07a1888f576cda679f8a4a5edb03d80ebe858047e"
let xnnpack_sha256 = "380e5185c4c48ede7cc0d0f0657ffb26df83cd9f55813d78593aea8a93942caf"

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
    name: "custom_backend",
    checksum: custom_sha256
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
