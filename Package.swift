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
let coreml_sha256 = "1d2b8d2a5805a699eb39f347977894d3af5dfa763b298b926016550e9ffefda5"
let executorch_sha256 = "39f19740a7c656d972e6082bae49583a6d4cc6396dea6ace2e4193688cef6225"
let mps_sha256 = "866739b76baec70e603d331ff34ff9f028202fef69161f63a35d2e8a0cf502e9"
let portable_sha256 = "6f761c0ae5651002e321bc6320604476ba0210f9383e535a2905cc1a74be55a3"
let xnnpack_sha256 = "ef2cb2145a466a0a9e32489497c7f4880e4b582cea3883158b7ae24427d8ae7a"

struct Framework {
  let name: String
  let checksum: String

  func target() -> Target {
    .binaryTarget(
      name: name,
      url: "\(url)/\(name)-\(version).zip",
      checksum: checksum
    )
  }
}

let frameworks = [
  Framework(
    name: "coreml_backend",
    checksum: coreml_sha256
  ),
  Framework(
    name: "executorch",
    checksum: executorch_sha256
  ),
  Framework(
    name: "mps_backend",
    checksum: mps_sha256
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
