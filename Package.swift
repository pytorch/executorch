// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0"
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
    "sha256": "93c2f34f9251d8de66277869fb86e5a803f427feff7c358bf281abf18408bcac",
    "sha256" + debug_suffix: "4110939b7af02ee901323104da504739351ae35c57915bafc58621cfafa7cb93",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9ed60273f4a45a1cf23f7e9732a5022e1e87f8c36849ea0ad6cbe82d1649c98c",
    "sha256" + debug_suffix: "641b782f8a5015d4790e80f2a812ade524842dcc79613239d2689db25c04996e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "9f32b42f22e8b4515f8e98c1f47f935a9cc952766acb57592d64b375dc2be34e",
    "sha256" + debug_suffix: "bb593649d82c776eaafc0741a9632df2c749c5bbeb7de64bc6a4d883b99a657f",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "6c7299a35370f14bd47bca9816d10f7ef28f1a1c085ab44bd6596bd5bc8b6d93",
    "sha256" + debug_suffix: "4517de56cb36089fbfa0597a087684a5689027557dcbf6d5475ecac81abd8a81",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "8ef95315469513e5af519bb8e852ff690e7254faf32f1892353ad752085f391b",
    "sha256" + debug_suffix: "a8aeb5b2cd5b70d9b15581b56e2ed28dda28abe4ca419dd80c72286de77cc4ab",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "b53eec200f6e149689b9815a9aa5be641490a2a3c5494381fb0c72c1dea0329c",
    "sha256" + debug_suffix: "ce24aca19ac5a1c75c92ebc910eb21e9f0b50e8d4d0963c5bfbc0a7cb500a505",
  ],
  "kernels_optimized": [
    "sha256": "da90dce7ea1a9851eb1ad9c90a1f8097e987725ec870c11de985a2c10b867dc7",
    "sha256" + debug_suffix: "fc5e25dc789b9dbb8afddca85f735326cd8388c5dd7e3e4741d667194ef01660",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "0b68c6deb347a4c7e3ba9b897c7f1b157e318c0550a967ec2430d36acd2a639b",
    "sha256" + debug_suffix: "9c3934cdc0016ce1c04fc1d7f258e870577e7be2526f849a6f5ab0e6a3ff6b08",
  ],
  "kernels_torchao": [
    "sha256": "44210621dbf7403c8f7300be905d4eb45d7ef886389dfedb4330944b27910fc3",
    "sha256" + debug_suffix: "c9b5d1fbe36fd151b7d8a6e5a06b28ba397fb38e8d077b3564d1707448c414ad",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "47fddb98aa0a49cb84b68fe7bcd0c80f40d94e806e65549a941a3c31d3539ec4",
    "sha256" + debug_suffix: "1b0b7ac189845b7dc44aa63f4810b1aac8fc57a82f25f16422e0cfeb47b89598",
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
