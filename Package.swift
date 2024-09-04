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
    "sha256": "127c15d7e7b2e18ec54a2eceb9e40604508d63dc13797ae69b995722069bdbd4",
    "sha256" + debug: "324f407a010099e51136bc4c5dc9c282bbe361ab8cab0162f7170afee41904d7",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5e8ab3a6903f355fc2807292a1f96c7234067f933c7b208c01297390f6c010a3",
    "sha256" + debug: "f5f644f38fb66b469b5b8e010775cafc0a9e8368e0394955e6da376e22baaff2",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6d6b7170c55c1cbed2e3120f27dd1f1b830aa9b81d7f7e87854d2167c58c1b26",
    "sha256" + debug: "1cad5ea02dd98941bfe9f92301f42364208bb981134d1b582e1fd2178c541df1",
  ],
  "executorch": [
    "sha256": "88de57f8a8bea3c2a6cca2c9f1badb0a508f2aae075846c26200309b12a99cc6",
    "sha256" + debug: "d7da0cbf47b4f48a56c2e7c6e1d791e429a8f3d368c0687a74720c999c008d8a",
  ],
  "kernels_custom": [
    "sha256": "b22c17add4c21bffa6e830c561b098a22ca99a7a71bc4e48d7151f4bbefa9bfc",
    "sha256" + debug: "c4d5a4baeb5e3d4ac53bfd3ff47af875cc59706af0a78bbbe93029b08877b4bf",
  ],
  "kernels_optimized": [
    "sha256": "5effe7a943413878391642701c26c6c79b300af0fa101eb4764b9c75a4e6caf4",
    "sha256" + debug: "535e52162f0d9b2a0166047e26b0f4425daca13b7b152acd95ba46d441477138",
  ],
  "kernels_portable": [
    "sha256": "43422bb6058e8fa7b7b06a7769ba94934ac550cf2f1a464b8cb8a93ab1a30b3c",
    "sha256" + debug: "367ebd181fd7c598812ac980a2494a2314ab532efcdc875e753c0be53b24ab89",
  ],
  "kernels_quantized": [
    "sha256": "aed5f2b5386fee0055c99ae78ce805f2412c7bebf0fed508a6c1ee08a2998b92",
    "sha256" + debug: "5a1a19d489afaec785eafcbed2c91047b4c84b2ce991e490a53d9190530075b8",
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
