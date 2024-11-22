// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241122"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "5f0c63e02898c6c40bc752dddfad2304679b5c2ce57618f4bed78d7760a16486",
    "sha256" + debug: "3298082c31f1f9c82d302bb53b86e1ab1fce23039dc68bde8a0ff4e99b275b24",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "bb002c6054db7c502fc0c4c642651e0de3c9831947510edd36eea9ea2cdd3da3",
    "sha256" + debug: "d9732bb9224529ce5775044e06dd53378302e6409c4f77af2e4279cf8576dc04",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "99e65fcabb7e0e3929976775af255cca0aa582ae2332c890c9153b9d0c059971",
    "sha256" + debug: "a52540ea31ce1998cfc70da470e4d40c42b6728a8c39dc646c2bbc75ac9161da",
  ],
  "executorch": [
    "sha256": "e1ef485b1c9be29b17914ebfb5207fc636cd7efb9c72c77d026785827e3b0ff7",
    "sha256" + debug: "a405a9dce2f23239aa08c2b2589f99c5f1592fcca34c714c5e14b3b57d64e44f",
  ],
  "kernels_custom": [
    "sha256": "6a9b50fec8b13f2d15ce22c6718ed31cf55745324370f9311416aa62d1ba4b20",
    "sha256" + debug: "7e93b0d20239318c89697213377a1bd0a2688eb1f650c049098bb4fb54a1a900",
  ],
  "kernels_optimized": [
    "sha256": "8bc0e8c42d8667e4c71b65e3c15921e1e0c4a6cc1e8f675d3a7deaaac2de3cf5",
    "sha256" + debug: "6f46288042720d511e9a2d1689888207a8bf8775c4275a47d9eb35c0993afb29",
  ],
  "kernels_portable": [
    "sha256": "eb3b3babd369fb4903d592d9f89e87441d844b0bc976b80841cfd8a4f194c29c",
    "sha256" + debug: "5816c141ef8e8a9f4bd9e4dcf9ffcd23f0644ce3f47a824de2194d515dc6c8e2",
  ],
  "kernels_quantized": [
    "sha256": "83362cce638f84351c4a272a7297803c23ec0cbbc5de9a0aa7b42a4f821982d8",
    "sha256" + debug: "ee3446aab790cdab9e723968cf4a13c4e28b71697d2010e4bc3b53ec989df11e",
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
