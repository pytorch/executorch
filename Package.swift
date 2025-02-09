// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250209"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "2f2028ccb1da6547e83b81a9c6f35e44a66e9d229feec048b6e3f72fd057984a",
    "sha256" + debug: "8689bdd8bafd20a1e9ec504b45f76ef6b59ab1b9828e6d80b4035a537b8c8966",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a2c0307b92eb413bdf469410cc9e55b942470fae49aafe78b0d1b0a8c9174730",
    "sha256" + debug: "293b5ed5b480f7c52e453383ecee1595e99290421f5e06c79fae64f1d2f21072",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1c30984198d1cc5161078ef9e59a14e6a53e4ed720e3069f6d1dae183d4acaa1",
    "sha256" + debug: "a20aa5086719f535c15d18bc865c4878442311a343a5fd5aa38aa5517f5cc335",
  ],
  "executorch": [
    "sha256": "37ad7d40df886f1c4a8b7f3685dc78067da22a440b80aa0bac5b9dc8a68e50fd",
    "sha256" + debug: "91b0746411f9f08ff12fcfa9def6da8109e63150e9ebec2cc8fe3eac1d135905",
  ],
  "kernels_custom": [
    "sha256": "20b7f1ce5936ee232daca6e8cb14e6676ef0df70fac455cb19c0b28e447d0d7a",
    "sha256" + debug: "083d03015589f349fd6ff463166949a7b06747983e01fefd2ca59ac86ba9e650",
  ],
  "kernels_optimized": [
    "sha256": "3159bd724776aff68fba431c56fb8d2b654415825d9090800e33a00afb251a35",
    "sha256" + debug: "34191ec05c6887a2381adc5e9d0865490e61df1446c811c0ecb5328a3eb7bf03",
  ],
  "kernels_portable": [
    "sha256": "10144eacf526ea79f18642ce91f7f3eb5eec25e892a86c1f7318145ead35e87a",
    "sha256" + debug: "aa765861e2743cb059cbe5217ac21f841494dd836e578d282b4df197492d60d5",
  ],
  "kernels_quantized": [
    "sha256": "6f82f0ef4be6eb5d2156d5c0bf365e259b85f30d09c80d10797ccf46fb1bf4fa",
    "sha256" + debug: "e9ca46b96f4fb877e9cb12b775247774392731df6705557d68a7d2e430d99746",
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
