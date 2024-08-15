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
    "sha256": "24f89f8234d54f5d195340941837c6d90def64458e0c20b8edba7b920a795724",
    "sha256" + debug: "48d38278af4d169c409d5605d87867fdc8dd29ea1dc40e66f126c356faaa55d2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4694b03a71ca679710792f09cdf988884c6801d3da91dfe07b39ff4d0da46efb",
    "sha256" + debug: "104e9f431b2566f6ca3c0ed2cb2c3b68da74d3892d6dcb0ce140b113f6770a81",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7938e73af6964c5c55b1829e01fcf9eac5c4c226b5d45e603cbd045ad620c94d",
    "sha256" + debug: "382c1b8c1cd7f550fa88928db5d7467bc2bc5313c3e7cd87b1509c82096d6388",
  ],
  "executorch": [
    "sha256": "fa9cee8333f41ef14666b8876157fb91144b57482dac44f5ec3b7079b5b5e07f",
    "sha256" + debug: "e4a037c9aa1a647fd7501b6bbc96a5b65689d1e86a4c518d24519a7307040ca5",
  ],
  "kernels_custom": [
    "sha256": "7c9f3636edd823b85050e4e9f8b920ec04c3b944c03b592db169326ed9bebe71",
    "sha256" + debug: "209781f9fbbc3752ec9b86727c97c675bcd6b1ca269d2740db5d672d1a29e968",
  ],
  "kernels_optimized": [
    "sha256": "bd690460de3c321b5692b3ec77f4d9c32ade45c02f461d6af37006a5804a1ba9",
    "sha256" + debug: "63a46b7e7c1db1c8fe7638592eb385507244f0cf8edfa968f5acee4364ed994f",
  ],
  "kernels_portable": [
    "sha256": "24e2db5d1ff1e51bfac5f58b605c67c2d2ab991d9c79a2368ae852928de506c1",
    "sha256" + debug: "8a24e4afbb08d981c6a658330b446a8f7dd8a900f5f90a790b7139a858298dbe",
  ],
  "kernels_quantized": [
    "sha256": "86c002c5cc647ee9b9dd73c13466adac18f2ab1372278505ce9a03c4344919fe",
    "sha256" + debug: "48f06f220102edb30e098a3a6611a99cb42e3bc963c73b87405117b2e0ca057a",
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
