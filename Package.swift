// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260316"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug_suffix = "_debug"
let dependencies_suffix = "_with_dependencies"

func deliverables(_ dict: [String: [String: Any]]) -> [String: [String: Any]] {
  dict
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      result[key] = value
      result[key + debug_suffix] = value
    }
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      var newValue = value
      if key.hasSuffix(debug_suffix) {
        for (k, v) in value where k.hasSuffix(debug_suffix) {
          let trimmed = String(k.dropLast(debug_suffix.count))
          newValue[trimmed] = v
        }
      }
      result[key] = newValue.filter { !$0.key.hasSuffix(debug_suffix) }
    }
}

let products = deliverables([
  "backend_coreml": [
    "sha256": "e18042848efb1391c9575c28494a22b05bdf3e294a8f4cb5c7c28c8278302511",
    "sha256" + debug_suffix: "a7dca74456a95bedf9be0f80a0d84f740a16c7a6d22beb9f4bd1872ecfbe3f91",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "17c481b9b94202b66eb68d96173d547d04ac5bddb74d5c6571abf17ed0bb99ff",
    "sha256" + debug_suffix: "35cb53c6d5c22f12613183c0a8bd2b6e3f717ac55609f2ea7ff0be4869cf5145",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f76953236889a4ad8491cba747ea1e6db64c0e67671f4e1788ba7c6c1c32a464",
    "sha256" + debug_suffix: "d4b1b84ee364a3e3235d5256da05c870fb5dc26174d095f73dcb56e9dd77aeb6",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d2d68e6a65b505fb993183302ee305ab283b5a2af218a7bfcdf0c6bd664cfdbd",
    "sha256" + debug_suffix: "c5e8c6bf079d566282257a6691368694dc66f67a02821b4601f2f513d2e622c9",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "644b2c9d29375ea80c59f7aa314155d7d71230cac8868265cb5da8beb0ccbafe",
    "sha256" + debug_suffix: "baa1a62f09321188508c04e0f83e8015a5a4f5d5798a4ca1894df02e6ef839ed",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "5881229322865c585a932914c7835043e71c1223cb58d78a75b47f1b015ff466",
    "sha256" + debug_suffix: "dc7b22d687fa542c7c03b161fc06a57372b5f4de51a2fd2aea78c452aa72ff72",
  ],
  "kernels_optimized": [
    "sha256": "06a87842578f8f1ee7cf2a97cbfd457473aa13e270a22015c0162d695c223497",
    "sha256" + debug_suffix: "d51b9286c34cfc099c036d183d00991bcfecfd672c21fd980246e46153bdb5a0",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "47a91cfc27be5681123a12767afa665e702f73ca5c4d6a1f551cb2387745f743",
    "sha256" + debug_suffix: "8aca7078dcd4e509f2b8917a3c21526a53f8b1ee4a027cff35e75848086311ce",
  ],
  "kernels_torchao": [
    "sha256": "8009c7da02817fd577b8ae1601fbd95109b56110681c602b5daf26b76da2ea24",
    "sha256" + debug_suffix: "b15b60a114750d292a75173a09bbc437551f883ee6f3a305c5e93e5a56d279fa",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "7e931f13dbd96ca7c88c846cd4101e0b6098d7be22a1541e77c5b5d15482b44b",
    "sha256" + debug_suffix: "16d76b0f4571a2ea52e77b34282d22a188d67741675195c027afd57a9861c631",
  ],
])

let packageProducts: [Product] = products.keys.map { key -> Product in
  .library(name: key, targets: ["\(key)\(dependencies_suffix)"])
}.sorted { $0.name < $1.name }

var packageTargets: [Target] = []

for (key, value) in targets {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
}

for (key, value) in products {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
  let target: Target = .target(
    name: "\(key)\(dependencies_suffix)",
    dependencies: ([key] + (value["targets"] as? [String] ?? []).map {
      key.hasSuffix(debug_suffix) ? $0 + debug_suffix : $0
    }).map { .target(name: $0) },
    path: ".Package.swift/\(key)",
    linkerSettings:
      (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
      (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
  )
  packageTargets.append(target)
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v12),
  ],
  products: packageProducts,
  targets: packageTargets
)
