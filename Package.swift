// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250118"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "965deac86928c14e7b9cded89285672b9ce34f2b74cfce688f9ebf6574496414",
    "sha256" + debug: "3594fbc56be0b70dc11e540657a8b898f64ed45d6f57a131bc7d461b8ab05508",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "560201e336c21c8acda5a05ea3aef647f48be86db4ee0e1256ba15becc5f886e",
    "sha256" + debug: "db193eafc239bbf564a95656d62f0f2f08bcbb178147fe3a4196171e3bc02d87",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c52f099c1c0659493800dff975ed1b083beb4152e6e36b501cf2ce43ea768b68",
    "sha256" + debug: "276ab43c2525a924bf771e6fe8b16a49de0ed62fe2d7b11723797d0e8f2f178b",
  ],
  "executorch": [
    "sha256": "018ca83ef71b65abb0adc3026c536ef8096370322cd4a7fd8cfdd3132c2a7a22",
    "sha256" + debug: "cd620616cd0f823375ab02bb59876e2ad3c6f17ffcb05cae5a2a9a460e54fab5",
  ],
  "kernels_custom": [
    "sha256": "26e1d32b29f2278af2515b0b1fef264cd96729696c2c5e53779d8e34104d30d7",
    "sha256" + debug: "7320b30adfd3d3b87396f0867c3ac60d50a968bcb49d884e832ea58ae3ef7d42",
  ],
  "kernels_optimized": [
    "sha256": "09073927c6289aff14309379af3b5e589180d6fcd1cd65213807a4da543ea74c",
    "sha256" + debug: "66545db3723f7bb24d8d2263af8995c29d53299c0a261263ca4bd5e1050fc254",
  ],
  "kernels_portable": [
    "sha256": "665d31641e4fe7f0ea68a1cb9054cfab7a2f649b2d53e0c0b4b9d05a1ae42259",
    "sha256" + debug: "18f092889c0404a786e6efe9745d4cdccbd21fcc6bae13e5630888b2bfea1071",
  ],
  "kernels_quantized": [
    "sha256": "ee2d517a2917a509739ee9cbbb0717f91fa02751b8436b23deef10c87e9dafe3",
    "sha256" + debug: "39ed31e3b0f8b0096e23d79cf26422f4302c177b41398b80f4e841b724bc918e",
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
