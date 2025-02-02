// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250202"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "387031cb21f6f171a80ba6683054af5a424922028cbf3ccd44ec079efe6d0b6f",
    "sha256" + debug: "6ae9c7f250e609ad0f2283e5718cf1ba4adff4e3d02225f5452daeca50969d81",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "18d2d1b936cc1926c3ae8b654ba798d3b34e2b6dcfb2202602199be9aaa2ba0f",
    "sha256" + debug: "a7a94bddb3333a07cac101c5cde83aef8d073e4b81e60afbc3b9e844c4ef31ca",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ef3b546e3d57c45dd6a486022159f30f8b3a41d43dc6dedf881613180a033525",
    "sha256" + debug: "72c762c6373fd5f80d5ef1f3ac7776a5cc526dbe02840c273f0c03c2c980a4d1",
  ],
  "executorch": [
    "sha256": "01948aa84b2230f59bb46b8b3358a1c6f6fc25cacbde83497f03185c098fcdf7",
    "sha256" + debug: "999b1b4bc6047937969072fdd2eb38773628dbf5f53babc1679b174febca4c69",
  ],
  "kernels_custom": [
    "sha256": "b400ad1e236a5c062fc33142494cbab22af0990b724d1374a5890c2390d629b2",
    "sha256" + debug: "d9b8b7a6c48f6f5f9b86082074d751175b7883406cba6c99f0904559cc2cb91a",
  ],
  "kernels_optimized": [
    "sha256": "3a3fd1c30cc9ccd16a379b6a666b32b1fa35697a44e6f1c89fae59635cc90192",
    "sha256" + debug: "b20947a4054a15bb7c99f89997e14d505ade5d0fa376fdfec18c1768f3cd8ae9",
  ],
  "kernels_portable": [
    "sha256": "781480171666c83499663db971a82a6aaaeb5a884cffec80ea64e091ecc2180e",
    "sha256" + debug: "1a5c77c7a6fc1cec53f48989607e62528b281f7a3700f4089938580e1ec73137",
  ],
  "kernels_quantized": [
    "sha256": "dbe3cc1f8e8895a9ec3072a11eaab27735430cbf7bd83fd5b42e36184bb00378",
    "sha256" + debug: "3cf3f106139a50ce3ca726f8acd19c6b4d2a7ca02a214bfca5fae40880d8ece5",
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
