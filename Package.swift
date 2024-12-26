// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241226"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "de3aadf8aa7cc77a59f8d09c735287c65291a2fe15bd55fddb7472e1ad9f65e8",
    "sha256" + debug: "7c607e010d0a20344ee8be9b413d34369d3035d767a8b409e3247d42d6467a71",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b060a63cd50d95e4ac1e2f28a6cc642277f81dc7ee722cdc5b9cd3ee0dfb80e0",
    "sha256" + debug: "8fafdca5be8fa45f047b1104d72b4f71bccf4a36c8effe160492a2be14a1343b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "54dfe39288a093f4ab886e25b191a259fa181c5828b91be3bd436ceea13b1eb2",
    "sha256" + debug: "7da78c606bd7be6cc87293c1262ebbf559fb7564eb51b9b1c0d8538134d852be",
  ],
  "executorch": [
    "sha256": "2698c02dd2d60b2a4674e0098415e33efe53f4697ac235e5ddea3adf0244ee8f",
    "sha256" + debug: "bea50faa62bf60546eccc1992e30ab94765d1d9c65850033dde5d1b4cfbf78da",
  ],
  "kernels_custom": [
    "sha256": "1ea16a8f9d2ee96fb29ebe7a7b41335faf277e7722cfa82edc856a5f761da76c",
    "sha256" + debug: "de70e5740371a3edef623a2c2e7d48979d73c2d0d35e08acae8a22a3ad20ce3c",
  ],
  "kernels_optimized": [
    "sha256": "52525191f11b96e38e3f33bad6c14564f291a930d6cb079c08280c9ab4fe4ae2",
    "sha256" + debug: "db0481b814866936143e6e3f6ea192dcfea744a9a5b9ddbe2dc3e258a06dc03a",
  ],
  "kernels_portable": [
    "sha256": "39cc83bc897153c6772e3dede2db3b6fa7b369d4b60fea7b7af02b8807a0b361",
    "sha256" + debug: "253a7bba189a46b5c445a2398da7af6dcdd4df006a4726861cc0caa87b96b356",
  ],
  "kernels_quantized": [
    "sha256": "471af9004112992692e04f073f43309732e295b9ececfa70d7bbd9c6206e2e92",
    "sha256" + debug: "138b8335195ea01c5c25084ea476afd633cf8629d3a619f5bdad1ed055392e4a",
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
