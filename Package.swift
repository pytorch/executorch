// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241117"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "12c3735503d082a4b305726727569b96dacf3b602b0347233900a3f6f02e462a",
    "sha256" + debug: "7427d8765cd61f8fec5cdfba71180e36d696224a503e581b12d4f475673f7909",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "fbc2d1b405459f2645d42460a41ac16fab77a513aa5bf36c5b16e99b2d7f6cb8",
    "sha256" + debug: "585fa5c2ba6fd809f2655b1d6abe6cba41777ed276d17e4b3239fb369eeead12",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "34d454e021bfb8f86358dc12c229a2fa84ab7d64a0c974e66d52e0715ea2be33",
    "sha256" + debug: "e56ab3bd062d2b7a4fb5e07c6c9cfd6521d5a6f7ba2d2cd83e9e7666f08871cb",
  ],
  "executorch": [
    "sha256": "064d91a81a4f4ac592b007c74aa89ba14a9d5a075cc93220c8e7a3e115baa5d0",
    "sha256" + debug: "4bad9022602f9977627903e0a50c5b573b38586489b1c070e2be7dbfd6703f5f",
  ],
  "kernels_custom": [
    "sha256": "2a0cb3ff1289a21f5591ad480d28dd619011c36c375d6f35099e175986346f55",
    "sha256" + debug: "0508248ab919f61a834fd74ee0e478c40394825e378451c06ff4638d979c0d96",
  ],
  "kernels_optimized": [
    "sha256": "cfeb59c89a149dc2e15437d91f59d4293e0f5aea37babe92f8b6ead7be6d8d9c",
    "sha256" + debug: "4a5d02e510e031cfbc0d0e247b6f56404a1ff4368e20df1ff69e367aeb9ce2b0",
  ],
  "kernels_portable": [
    "sha256": "1eae3cbccbda66c4b2461cbb7a0995d1b79ebf3c9379a09a7e1478d37b19349f",
    "sha256" + debug: "80b7aa08134bd6e6d0de2b18cb303c1d9cc43e4426e2c17a4035f5a2887c5b0f",
  ],
  "kernels_quantized": [
    "sha256": "8ea8e5eedf5ffad6340b6045b7537c0341748cc85afc7e058370bbdab59ca096",
    "sha256" + debug: "c38e0b0c7b78ef4991c30a4d52ca40f803a6bacbbcc237f11f1d833263fcc46d",
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
