// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250129"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "c380d497d83bad3784ff146b9d5dd2b6cb33ea3ebf0fd5027cb4b19830c856d6",
    "sha256" + debug: "d532b070b4d70647d1bdbab440172a025c2b58c95b42be3adb123d6bca1b1a95",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9913c00cf0854d8702c168821f0bc9f657dad9be9f010d26e354fa49ba3f3a16",
    "sha256" + debug: "3f4f8949be709609e755e7fc9f3191c58de9964e11c2597614aed064ed59c062",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "9cd21cf7c093d560693497c68b5f2cd7eee9e968f524ab2a6d0cbe40fa78e9a4",
    "sha256" + debug: "0cae7fe4b0193e56e17bfe88b976c6fef5ef695a6c750b635673a113873f585f",
  ],
  "executorch": [
    "sha256": "e16331c6236a0f36c74525560a07848d1f191155b207dd4a087b6a5f2efb7d11",
    "sha256" + debug: "588444331ecbc53ab82a96412e56d6d2c2a3cec3b33b6ef74766b321eaaf26d3",
  ],
  "kernels_custom": [
    "sha256": "d7dd08db8d9dd9386102916fe5325f180775b883076e3784bd366f5f0cda3b7c",
    "sha256" + debug: "cc5ad9e5e0298d38d720074e79f2c0fd2d5eede503f43fc5120bd323ebe98e05",
  ],
  "kernels_optimized": [
    "sha256": "62ae74510641cb07e969b17f8ae893ddec36e9de4464488af0bb9b779e2cbcdf",
    "sha256" + debug: "bb3f9f66f7a415c0d0cc66b88ab0048f21cdb558e7b85a95cae9cef0facc230b",
  ],
  "kernels_portable": [
    "sha256": "27de54d9c2e4a03f8ca1c5743081ced87e7ad3d1000471d1d35846f157fc7b9f",
    "sha256" + debug: "2a62c8ff7400df81847dcf1991901b82affbe6f2707e6f933408f7f1d91675b1",
  ],
  "kernels_quantized": [
    "sha256": "7744fc3ebf8ff97587b0e765141800f6775b68d77569ce0c434f03fbfdb1e101",
    "sha256" + debug: "848e7cb34de3e9fa2117a5344475c9c35eaba0fbf3883265af6cac0e7d6e5781",
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
