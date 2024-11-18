// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241118"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "10b3a6a4bfb299e4eaf7b3e1a6b190e62dba6396c7412cddd61c954bed6c7c78",
    "sha256" + debug: "a6b95a375c9fbed3f1bfe97aee8df4ad54e470188d81f8683f9eec9e1575d959",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6bde100fb0b12c7ed2f750a8cb4335528343e40365140a869a7b893e575e9dcb",
    "sha256" + debug: "b0b1b0543af9e8ea4199c52033bf1e4d2f002ec760e2fa03b43d6bf5f219e3f7",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c245c088bd5386999895ae951711c02ed4425ca56158055f5831f68b94bdfb12",
    "sha256" + debug: "f52ed144fd787beab6e13c98ddf0f7c5cc1d6a04d0e5cd286825af8d30149057",
  ],
  "executorch": [
    "sha256": "5e5d35d250c072b877cd77af8c862a6c43745e1c5e11a4f4c088004205400c4d",
    "sha256" + debug: "d933bfc40bc6994ab944bcdd0aef5b787066f9019b55e159b2299f4add8571e3",
  ],
  "kernels_custom": [
    "sha256": "0945a1f002066b3cbe8a6d948c2a3eda083aa575f41d889c57557a480cc95e1c",
    "sha256" + debug: "d5ee79e588a6d8be39fb4a1a5ec2654d589b5ca07c9b31ea6506a67e42149682",
  ],
  "kernels_optimized": [
    "sha256": "f4340e428a4548b490a27b379c9b653fac8ea64938850d1d2d6b0c3c42f5fdc0",
    "sha256" + debug: "a6016e888b6c3a132f31d6b6399f2023e63393b882c87ffb044a2362bc7f1e10",
  ],
  "kernels_portable": [
    "sha256": "3ae6e6c68e5c72b23975af26bc58bf640e7cfbd302191043a09f665d2a22d8d5",
    "sha256" + debug: "ed45d7ffe6656a974597fe54b3983630374ef7016135b96da2daf44e00c8d42b",
  ],
  "kernels_quantized": [
    "sha256": "47f3567d056ba754ee8c815120961b3a56a7c1c0d51fe8b1dc5e81cf2785bdd3",
    "sha256" + debug: "d49a55e3e19226d138059259fdd9290383e301f74da81934e14805d138badf07",
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
