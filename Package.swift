// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250120"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "5e25c41e8bbce9e7b70f116bee2b699878e5240d839c00d9c29743d281edcb12",
    "sha256" + debug: "1773ba1e6ced943892e18ab5912437b8f6479357a6e769ed1faba3d78ea0120d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "f6f25d23fcae965d2128f8f3d4bf6e738fa14e67557ba418d92c260923acb3fc",
    "sha256" + debug: "45b5363eff82c4fa22e0ca5c16e0d16b993e27b253b29e64dda6ac06bc848ebd",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c2b618b9268241bc5573265d6664138899e1cb1b105744cee4fdb92604c3f19e",
    "sha256" + debug: "d6e4ad7ec4c6bab028ee9ab9de71897d5ab422f46f4eaade2cf299775250d00e",
  ],
  "executorch": [
    "sha256": "47c472ccf0748988f76a00d172530cc49b6a693050b597a78d00950b623b164e",
    "sha256" + debug: "a8f1d0b7f0a67ca220b5d9ffe5ac60880be1dcbaa90cc9c97445091a461472dd",
  ],
  "kernels_custom": [
    "sha256": "c4f8cde07c29b36817ecbc264b2c9b49b814cdd571f76a081c24186caa544513",
    "sha256" + debug: "66fd6c0ff5cc58ecaa9a563ab4f098a3968f83d4f3dbacc6487a2e3720796efb",
  ],
  "kernels_optimized": [
    "sha256": "2fe907dee2f33cbe9a69b2c26983fd14b8da5eca617c617e9314d19d09aeb4c4",
    "sha256" + debug: "1066d984d8f623af115b5b560b047b3035b8df33010636b2b89d069680cba3c2",
  ],
  "kernels_portable": [
    "sha256": "fe2a1f20adbd37a4e2c1cfaad26224fe9644f495c656d9a8eada5c5bdc99dd53",
    "sha256" + debug: "fea3ff14c08fedc80b60e1f9faf0e4ee378b965c93a64e0eeabe79c10b7796ff",
  ],
  "kernels_quantized": [
    "sha256": "832ab9a0f42effadfbbef998704c4dfaf711fbcc83e01e73939aa6c90d40e999",
    "sha256" + debug: "e61e5c7e127bc44013b0255bb09b867247a5a6facdbb75aab2e041f220e035a5",
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
