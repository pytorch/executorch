// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260811"
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
    "sha256": "ca91a06ed3b05ef1d6fc00a3cdf16b6cda56bd99d37a016357a051515e2b845c",
    "sha256" + debug_suffix: "548113456f0b213a8f2d07a78a9af456978643ba9db160f2794aafa480644404",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6ae8ec5784630b6d4eaab9d99302c41009e9164ad3ba89991ae443b85181099b",
    "sha256" + debug_suffix: "c4b6fe8dfa83f0f1546acc228073222b0a30c37c0d01445c51955e731aff4469",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ec4657dfc91f306e1207f0f70ccd3b4e02c121228a09605c3e9816f7645920ca",
    "sha256" + debug_suffix: "63709608b55fa0670ad297799f3b25c05a60ba845c259260edc8b663ed5a4873",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "b8c3807a2cf61c4f27fe05637dd8d3e82a064f4a5c06bb8d4b8fe53981da0cde",
    "sha256" + debug_suffix: "81c35d57c0f68c97dab90646b835404a136ff4801971cd9409737283a4eb7885",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "d7302bb9981b12d402f6770532b4270330ab3694ea4dc0bcf69c1e67d92f121b",
    "sha256" + debug_suffix: "614a6e9cbc4a1cd40866e2df9f134ebe9957fb1ff639ce1992e69d0506ba2b2f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f6bd73c29032116941343e5c9bea5c2252b79ebeab5168502c5be51403a1f92a",
    "sha256" + debug_suffix: "73b40fdd5663bcc3cd63c771a161e1864fa7749f40d0b4cc90f8c7467dfea767",
  ],
  "kernels_optimized": [
    "sha256": "8ce5bfdae38b9f36216a10e97af32cb75e6571e7272a9fde4e5d6278b4336ad0",
    "sha256" + debug_suffix: "70fbb7ae9137f08eef613bf606d2ce85f6188a16f220d458cd529533bf2782cd",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "bc20b584de82dc1d3bd162a3f4f47fe5ceab98666697d137c26f5a125836bc92",
    "sha256" + debug_suffix: "c4b300f57f1fc41b5ccc2b7d365e5f19bbd324e35a9ddeacc3029e01d9ca2d00",
  ],
  "kernels_torchao": [
    "sha256": "cb79a112c782a32790d831a44ffc97b17399906f29f25c597583d28ead099096",
    "sha256" + debug_suffix: "2904a64043e97d11c9f9180397cfb73129b915a39ef624d7f7c1ca50c51b5429",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "d8ea5e29af7af1a062d07b9c7d1fd7b9f9f303e84b75fb9bb1de011f77116d77",
    "sha256" + debug_suffix: "a28fa8f85cbbabad7f109a1481d84610b6a7a6848be6d08a1c8e366d02ea2e38",
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
