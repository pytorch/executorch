// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250222"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "3f4921ff82e65e20743205da49d4edcc5263a2e1c33e673c333e7bca78873631",
    "sha256" + debug: "3deee9b04206be70c5186d698917fa115f36a0bc173e85e99b03844d078cf111",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "cad45f33372f176b3c408599d762041d25a1364796626f93b03a6fb042bc0872",
    "sha256" + debug: "727db2efa06fa32b49590f7fac5240472ebd568533d9c8ab3fa9818ac92403c0",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4aad32eed3cb2e46b5d9d18ebd6bd96dd71c9171c7dda6c7ec395096532a96bc",
    "sha256" + debug: "8770cf2d250cc794c145f1e249141ea4b228b72bef2481c82fcf39ea119c31b1",
  ],
  "executorch": [
    "sha256": "7202a2170fae80843af96601a6cb3fe0837e0b60b35a573e7384a6e94ef8958c",
    "sha256" + debug: "31831060f2fc2186103540dcafe7b503f39cd8971da1e3993ac120a2a3e0984b",
  ],
  "kernels_custom": [
    "sha256": "0f4a47e67f8942bd54d912904ade5c59c13336f17f5978a18520d21f4ce74892",
    "sha256" + debug: "e2ebbb9a99b1bb3f0e0f1247f4125e6c7b75ca92e23b794b0b20a46daa981285",
  ],
  "kernels_optimized": [
    "sha256": "4b5637f4725f4066ff7adf66888420d6b1e813baa92628fab59bc35ff803ae87",
    "sha256" + debug: "584c4c26ca7c7ee9514e682d910c2b0eeed481d095dd9ddc89bd50aaf7dd9cdf",
  ],
  "kernels_portable": [
    "sha256": "dbd0b6220074d2f849b13336c409dddeb93da21120fb9cb5221f4cb686de4c47",
    "sha256" + debug: "fd7c1293a3ed0201e2c5fe706d072c391978a89c5dbcb3bc77f0bbae418b0a9e",
  ],
  "kernels_quantized": [
    "sha256": "7666a656bda6f3b7e3d91ecbb480ce72d3eb6135094df0e3309692c687f09d33",
    "sha256" + debug: "7c552c998098ac78a4ad1850137b627a25a24480250fc9baeb5bb582e7a10518",
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
