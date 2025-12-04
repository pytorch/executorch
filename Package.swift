// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.0.0"
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
    "sha256": "9b92eaf2cfb8a51c55d122af44aea1a4f6c001001e2c93f3285f8d242fee4fa8",
    "sha256" + debug_suffix: "24ee7b46e723695526fc9b6c2147b4ba51f6d1ede4916d8ff51a80bc51cc2f11",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6fbaaecda52f74dd01ecec0adec35c6d6befccdac20c5ad98d000c85f66663ea",
    "sha256" + debug_suffix: "4ef6fe4884b95fea2da352fc9520324589908bf8eb516b95e21b659e2ee6ca01",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c8cab91ce5ee4cb4b25e03f1f6870ea23293f62b98e046adf2176c63350c3fd7",
    "sha256" + debug_suffix: "fa6b3c69325f53e5b4fa3a9d44856fc91dbe45790eef86f5a2d4327840499d1e",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "1c5fc01ef54e750f41e86e92767aa228c59d9ec9beadf997b1dac77126fc8b25",
    "sha256" + debug_suffix: "ce9569546ab2c8a9e1a2f422e3c9eea304c019d0b994aaf9e24fa70c7c28138e",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "352d471147112b30b9592073a37f164e4f7233f400b17bdea0974203cc693a90",
    "sha256" + debug_suffix: "9b1da80601fe716b3c7ecae9706a09bcfaf7bccb222bfb592ae5da7f64f03e07",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "5b1c3c3a683af9c8b461b29efd94c4bc2e4e374be88d3859af448ea7962858ca",
    "sha256" + debug_suffix: "d3c9bad74936ded8343d27b37f9dd728662b3d018af50cb36b9b4d88cc9600ab",
  ],
  "kernels_optimized": [
    "sha256": "df38a9748194ac0226e09736e75a7bf3b802ec77a0cbe4c51c7649278fa4ca58",
    "sha256" + debug_suffix: "40ad7347708e9bafea2df7d810b55364176baff7ddcecd94023474ac49171ddd",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "eb763a3a52f19bbead9a369a4a9e16858ae48f3520824e3c606030f0d7c616a7",
    "sha256" + debug_suffix: "40ce70b69656b7b7ed3f64fec18001925cd313acaf14176214643e4263507ab6",
  ],
  "kernels_torchao": [
    "sha256": "8400a6ab59a7b962f2784ef853072115e7e8ab0b6f9a42efb26ca0a74baaaa84",
    "sha256" + debug_suffix: "e3ffc0f50c990a04a87d8ce65126b408cd6e1ebca8c0538edc3cdaabf42cf48e",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "c18427a9342f1ff48b99d60b2dedad0bfe8219d029708017e24ea8b5853c3348",
    "sha256" + debug_suffix: "a61201ecd489d919f5c38dd354c4198322cc551e0f31c1a7c8ce6d590087284c",
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
