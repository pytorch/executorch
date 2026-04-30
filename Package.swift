// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260430"
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
    "sha256": "679280539f2f6050bc96079cf38d9186198e3aab455ebf5d5e72ef5952403ff2",
    "sha256" + debug_suffix: "cc4d1ea6dd231c3e1e3ae4831e01937f6a4697bfbfc52647870f77a354d49756",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "26b9b42c824a91ab8ff66192a0c1d80f37b17fa71607971958143df064e2cde1",
    "sha256" + debug_suffix: "2649317a827f0d424bac5242b3c25e782c686af57c402f79f5ce9ad0e3c8afbd",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b22dbc44feebb9e98be2d4f8708b92a4e3950a8fbf574f8781a8e518a2a7686b",
    "sha256" + debug_suffix: "40f99ddf9281377d6581836ffa38c95bfbbfb9429f829e853fc4484ef4a49e69",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "eb2973b347af8276df4f58fc01d74f52679101c0d76f1d83b5f69bbc1b20c392",
    "sha256" + debug_suffix: "228860fe6e2fc290dc3154f30d740204fd2c646470a1989fca51d0647754ebcd",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "c9f78c633d9aeabb124872323f723d3f734fdafae8fcabc054e8d2be33ed51a2",
    "sha256" + debug_suffix: "f02bef53b52c5566784918351305a47fa1f51be8c3217c4b899da3a6584a7382",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "b6be7766ac15b721035292e8301ce416224965bc657ddefa1a758ec911dac366",
    "sha256" + debug_suffix: "8d1f41e52a3e5b64d891a6eadd81a4a9d7e1f04e7e594be3a354f8a3620eafd5",
  ],
  "kernels_optimized": [
    "sha256": "913e01f4f4cd225b9d2f7778377a01d36d408204b7f9bdcd10f2fec8b67e2e81",
    "sha256" + debug_suffix: "f24ebcd45f25f79dc97779991c1971aaeefad3bf083b2120fe752e48547ae359",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "e78d62d7e54fc11158a01c0c47bcbe69dd6b5bbef7ff09dd5f547f93bfdf38eb",
    "sha256" + debug_suffix: "5cdd9d9cf9752442589663b63ed3f7a6a911140ef9e4c1b98e2f9785f8e7ae8e",
  ],
  "kernels_torchao": [
    "sha256": "743b3c4b529c2e92b54836120792eb5c04b02733cbdb0e138c6b15196459730c",
    "sha256" + debug_suffix: "ae3a5ec9b2c58bfad84fee9de7e3eed2e2f400697d38de3fe3f71005c4a17538",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "2b7dbce4e4c08cfa2a2d534106ad586722f89d182d9c7479b08b762f00bc314d",
    "sha256" + debug_suffix: "ed5ac04bb0d101e6758cdc904ba7a84082d0d8d15161dd83be6fe45a85ae5de4",
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
