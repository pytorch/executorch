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
let coreml_sha256 = "786c6d621e402777fc254ee403558539585595afdecbf8df743ca4e298adebf1"
let executorch_sha256 = "2ce568bd2704a5a7d7a0d4d6ed7df9e291f741b17f48767f0b3bc2a40ce797a8"
let mps_sha256 = "7b5a7805346caa5899c614573cf0ce715e2552db8f42c8af6b353f7ebb73bdbe"
let portable_sha256 = "52b7e86f02bf72eeaa3295b0612880376d5814cbe34898c805225ceef1d7bc6e"
let xnnpack_sha256 = "3fd6e4e1d9687eb25e2638bb3dfbc429b736cbf47e7ed769f1dbec62254e4cdd"

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
