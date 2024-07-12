// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "latest"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "76c884f222d72f58ad805b50bec5bfe4372741b727f13eed75c8ae85238945b1",
    "sha256" + debug: "fbd97aad335b9280109cd3e615a2c9f97a99f19339e6fd2d6122269ada002c1a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "13396647a9e65f4b2713d62af8b603b1d77fb2c3795e2a904a7347e24345c188",
    "sha256" + debug: "ca1d7850685481f900e4f54432212a0e7981b09ce3d7a4eb9eeb2600b8e9f635",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7eec0318eeb1bde9cb35a3f3718e4a7c9a64483e90e0228ee0ec50de1e9e9ffc",
    "sha256" + debug: "e93fabd9145e20f068e71a84406dcd18de46546963c57d9c1e4331ab4d2f6734",
  ],
  "executorch": [
    "sha256": "f33b09d081eea54eccf16e0675411bdf65dda6042dd4c94dfe40cd6bf144d3e2",
    "sha256" + debug: "1d1fac80c217d4a78079ba802bcabea11fd8b88f2749a800d17c4c2245c237b3",
  ],
  "kernels_custom": [
    "sha256": "2785fe6f4aa9237d3dd42067abe001e2dc8e75bbbb7de2b9a4a94ffb6d8b1924",
    "sha256" + debug: "f42da94115eb8551ae3c4a0f13a7cb0dc161ccbe0882a8cc2d0c1838d4dc7d16",
  ],
  "kernels_optimized": [
    "sha256": "6792c2ba5b693afec9adc4faa5edac714ddc6ffe6686d7d71aa4d25662899dff",
    "sha256" + debug: "5e905095a3df7f2da1460c72cea1cd99cb1045cfeb136a157b76dd5691296216",
  ],
  "kernels_portable": [
    "sha256": "08d9c096d84f4422e3670483208e24b0fbe17acb1896b252fb90447da767ca70",
    "sha256" + debug: "ccc71bc149db8961c30df42279f20571dcf9c0194d97368b02cc9317552339d0",
  ],
  "kernels_quantized": [
    "sha256": "991dd0cc26e6d9ee0b9f651c696599844157fe400688113573c16650213af33a",
    "sha256" + debug: "cdd4577f74b17cbbf6ddabdd9c0d316fd837fe33722f81bc1a5971c43b54c80f",
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
    .iOS(.v15),
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
