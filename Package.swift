// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241214"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "318869dce0cf62cdbdd2b406306f75689ff524b8832ccd7ae1a49faf46eee968",
    "sha256" + debug: "56b2d2a97c4a97e98f18b380d23300fab699152c798c336aa8a943c0aa58d633",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1b5fe1e99a7ce980589a1cd1c0b68e450324129036332940f0fded01adca68b8",
    "sha256" + debug: "a7db769e197921aa0712666226125fe1fe57cf4c053ecab479f167b1d2cd5b05",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a48796ca9fc839de9ffde03d9379883884333396909234142db3df9d1da50588",
    "sha256" + debug: "4c535483bdd8457123e4f10cb9c36933c6c3f85e0ed1b6fb7642aba51ab89835",
  ],
  "executorch": [
    "sha256": "7ea314cf656be56f606ab566eb509f5b0317ee620835bd9d566c8052e3d10b63",
    "sha256" + debug: "7a22a55248d1232fb36bc75dc8beb77b25393dc84c644b783fa39bfb44129b4b",
  ],
  "kernels_custom": [
    "sha256": "a7e51ee78eddaf0da259f89278b4c17e2ae601ff787922a6e050ddb19547939e",
    "sha256" + debug: "34e6cf797e7fdf18ebfc0bd5dbb8772aa38d6365ca1da9627ed788c671d38883",
  ],
  "kernels_optimized": [
    "sha256": "116caba5e6b1beccc372b3c8ae4948a3172033ac0d5cf0d1df9cbfdee5102186",
    "sha256" + debug: "d99576ba22b4802f6148730a76b47e6a6147739f8924fbb9f1a5825f9d8d43ea",
  ],
  "kernels_portable": [
    "sha256": "f8ba0ef464e19401327760fa79639d4c3e3b5b3c060e302aa362c99d316d71bc",
    "sha256" + debug: "7b315ec1e98cbc6feec8821b8cbfb08496e69a1d115a1d6432f37859026b3ab5",
  ],
  "kernels_quantized": [
    "sha256": "9c83a57d04d77fe45c4e10f121977685c4d147778b7165697e1cc55b24702c5c",
    "sha256" + debug: "bae218dc4e197143019f3bf698dc8ce512d431e235046f8e48781a2367d00eed",
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
