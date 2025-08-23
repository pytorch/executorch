// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250823"
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
    "sha256": "94a7a9df1f4274e81ac30902b39fa90140459b836a2ca9dbe635961c1a40298a",
    "sha256" + debug_suffix: "6ec4409f4aea3a1d62e874989ed3532763d350b7560a95f8f46e7b43f5553e01",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9661177b75a348a93f7d22b49234aa19ae7a66817fda0c0fcfaf615c4e54baa3",
    "sha256" + debug_suffix: "9b3acad3e74a78fa1fbcd18113cb17c62399dde05096f56025a71f65f7c88aff",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5a0de9603bcee3f94004693c157817ce56c1c5a51d2191c6960b406779d59641",
    "sha256" + debug_suffix: "799c55fb00c8234e12b742dd153a1464cd8d1c7b8892b4d113272fd82e849fb1",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0d9aa671e578eac7ba3eedb624f7b3ba1b8e9b7233ad01623270ba41c7a51a96",
    "sha256" + debug_suffix: "0b35c286d433d59a2370353e33820089eeaac62f9479ba43aedfd1e40415701c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "2ac14ebb44d1d32ae08a3b5868c019e60cd523d1ec3932d96880a24584cfbaa4",
    "sha256" + debug_suffix: "da6ad6716aa8d411292c77478e917ebaa9547dbecbc26c6d5de50ae8dd09a5aa",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "98060671b35ba1e4e912f968010ea35eaddab4bf38d8ebb1c3126602a49d9322",
    "sha256" + debug_suffix: "ae504ce8fc977f566e57515a02b0a44ce138e52f263f40fce40d17ed9184a909",
  ],
  "kernels_optimized": [
    "sha256": "dd5edae7e543ac34c6bf4dd45af13fde7f4581313955a9d49ef98011bc52edb8",
    "sha256" + debug_suffix: "0ffdd201bf0c0bc94f6879aaa75eb42585610641a9d60bcf5431f9110ae4d675",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "e1b5afd039fda3fa33494588df9487499e6b054f408ed250a46833e2f3dd97a8",
    "sha256" + debug_suffix: "0e5ccdaf365f82415d7e8419ffdb2c49f554fdbe577d3aa1d0e44210ac4e565c",
  ],
  "kernels_torchao": [
    "sha256": "__SHA256_kernels_torchao__",
    "sha256" + debug_suffix: "__SHA256_kernels_torchao_debug__",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "852cac1d894a242c7685e02dee158a0fb7a450d65c3397d4a2790290616511c5",
    "sha256" + debug_suffix: "3548a714be0887cd7658b5e7d1ebd6c1dedd507d42731aabdb45a2cedd1781a4",
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
