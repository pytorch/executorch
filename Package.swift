// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "ced887c701874343eb14c75f402ba0b93bbe5562f8bdbd96ffea37d115c8b0c0",
    "sha256" + debug: "df5a13c96153d2aa172fdf75079410efc12d53a48a40b94ba017fb16ced9353b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "33d780c61aa9a07bd7c055cbd081edd298e9e2dda4a38c4d8c10639c4a0d1038",
    "sha256" + debug: "ae7e7164988ee4ca4b3d8e5edf15e3a34b5fd95b8e1b4cafc46fd31abf502141",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b41aa352d6524b4e505ad6c9b3f39bfc906ca92743a7ca8fd45a646cbce36783",
    "sha256" + debug: "7ea33349b07453a953e54f699f2c5b25ca35d0ffdb4e5fef13a07658a9807150",
  ],
  "executorch": [
    "sha256": "a770d660f551ad10a6ad856e49a9fcc1dc63527aba2ba04bba6cfff050d791bb",
    "sha256" + debug: "4c813728687eec599f5c338adc88e2f30ccbd2a905503fb2739f231f7d4f52bc",
  ],
  "kernels_custom": [
    "sha256": "e5e9ef2b3bf65c95955b5d2db8f8ac4073a5bd0a4c262fd2a287b214ad629719",
    "sha256" + debug: "4a4637344bcc91f04233b8dd2d5161ee31210f97de7449f59bc0c291e82e7b06",
  ],
  "kernels_optimized": [
    "sha256": "e102de88d2e8c730aa4182e5984dccfa6ca369395fd10da97a5f260f8213ad45",
    "sha256" + debug: "4535888046f72388baa1cee77d43a050e72ad4ef6898619d66a6b0bdcd6980ee",
  ],
  "kernels_portable": [
    "sha256": "1aadc3c1ca72c21dcc0a7c238ae9b93627d7b7da7caf1ced8567d7870729d840",
    "sha256" + debug: "b7108d0de33473202af3958235bd8e73e4330a7465b240ebcf4047088db9bd06",
  ],
  "kernels_quantized": [
    "sha256": "7f479fdee55b1c330ec880fbfc53ac73c5ea2caea67477cb33baff79ed583458",
    "sha256" + debug: "316392d2baaf71aa1edf26161c668d3d0d79a34e4448a8131e5b663450eb3718",
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
