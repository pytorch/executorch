// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250428"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "d299c645b568e24d8da4c579785706ef6ca0f9610531810ce9981e7b5efe3888",
    "sha256" + debug: "5e08631fe37ef24a0d5cf2b633eb65a750e8200538cae640f893600ca1e9d3b5",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "fff7d26ea43b54770db1e4e62b62e82345bc7b857e63bb1bd643e75664d8002f",
    "sha256" + debug: "2366f207386d80d22a6112d396458732b899d70c2706104de6c0d2c7ef4e17ec",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "101cf01f45340be00e4711c7b1638f6de36d02b6c4528b2d433b7d6fece5c639",
    "sha256" + debug: "deca77b7a56d4085408d87dd57af4b35b4f0212d0f3f30d9020ca3569566294c",
  ],
  "executorch": [
    "sha256": "da0a4dae0a32962b9d9a61487e321d5bde4a031d35489fe04b85f6e2211e0856",
    "sha256" + debug: "cb7555384ab81b86e0993752b4de86956d07f19dc0ab788b343a0c1148d48882",
  ],
  "kernels_custom": [
    "sha256": "ba9e7d846e2866655d1e47bdb17407fecf4756851e71149dd5022593e40f7107",
    "sha256" + debug: "9329cae01fd1e3dda8cf117281ae24a6c23433384b6c8fea2d49289912a1e83e",
  ],
  "kernels_optimized": [
    "sha256": "2ea1dcb20f8d5ae7572ea5dd29a645f2cad1b5937212a0596ece1bb6a71fff4e",
    "sha256" + debug: "cda0a20e93f7e65cbe5446037b02acc1b690b2152052ef6fa735bee77c66d3a2",
  ],
  "kernels_portable": [
    "sha256": "13848275ca79fe078fa8b31bc1ef4b6c71517f892864cad5b550c0abcf1b308c",
    "sha256" + debug: "2294789b0dd20e2eca2e770b012cb94c9f04f5a6678fc94e4087b60a4d909cbc",
  ],
  "kernels_quantized": [
    "sha256": "6a4c6f7f83c521c9ae2ffbdd168819b4eb8ee1ed211f4818eee4067d403595fc",
    "sha256" + debug: "967241d67edef1882093b2e84b7b5290b12f57ca96276d20221ccd2f7633013b",
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
        path: ".Package.swift/\(key)",
        linkerSettings:
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
