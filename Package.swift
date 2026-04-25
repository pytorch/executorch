// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260425"
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
    "sha256": "bd84bdfb0e13aba647ac5d4c1b0c5d3547b0e7f9fa16f30a4a1b21dc5dec57af",
    "sha256" + debug_suffix: "bb41c0f67dc198f35e049d4c7f4d6525f638784e97e8d913bf0ef573361181cf",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2a364254c553f6ba451e55d0d2aa5c45a1d671db1cf9f7f0d647e38ec09a0444",
    "sha256" + debug_suffix: "8436529343c9abb6d82cd43cd703ad18191fb4c06de015b1cdf2bb44b5a36486",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "dea26104d57dd6759502c0838ab79dbaa97e09010aa8cd80cf8abecbf0b184e9",
    "sha256" + debug_suffix: "5b2294dc2bb051ff597005b23424377fc3e71229032712042c188fc48e0b320d",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "a033607ee1f8c95516b1982f91fc2165a6977660b46f56866ca4c3281a75847d",
    "sha256" + debug_suffix: "db04661829355895e9160650845ba98b3254cc60c968e951f2ff0dac2f167114",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "27649e788996c65e0bf0d4cc01d8d60080eb6ca82a1d0b4f40583c7054accc1e",
    "sha256" + debug_suffix: "f3c8948173561196e3ff2407473184ec0cba6aa93d33d4951113c543c122045a",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "2eacacd22d12e46b47dac6449a2ba59a07bbcfbb35c13de8aeb127d393999bcc",
    "sha256" + debug_suffix: "de9066599558cb5fca3d71025aed3cf5e9b1d6e8d564a93bdab96db03c2f7334",
  ],
  "kernels_optimized": [
    "sha256": "8322c105048422b0490bc413a66b813b8b3775f864751269bbc6d1d6f46389b7",
    "sha256" + debug_suffix: "f6e4afae6c6fdbc8ac3336417b986ab1f65aadb33bc2f705add5f6f6a457b59e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ef0422ffc834e4d04d2636ffbb930514b8530db2a13a62fee4732cccd083aa5c",
    "sha256" + debug_suffix: "1ad294d55ae5c946543c8893e5bf8bed1b59597e01f4550d4e56a1e8c527b2cc",
  ],
  "kernels_torchao": [
    "sha256": "ab4139ca21ea74f6e23b301533bc129630507dc38d71e6d5edbd4adc41d613f1",
    "sha256" + debug_suffix: "ad5ca38f9d2406c981709f9ad7f049635817fed1d3af73b5a387343bc2c27005",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "0c482289af612cd5e2efdb59e6623da9abcd35275b0748660601f1abb638b39f",
    "sha256" + debug_suffix: "92f4764ef185df9e2090e207964ebbee676fddf8f8791b41a550c60455c58082",
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
