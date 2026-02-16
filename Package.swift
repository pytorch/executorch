// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260216"
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
    "sha256": "27031bc113298b4f618289928533e6f702e6ffeb1ee5928927dd8427cad20321",
    "sha256" + debug_suffix: "8d99747f1df6d0d9f430881716ea8d7eaac5744431a2c4227da0bc29023cf717",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7a93c2b676cba9a5bd5c5ece47b58e886d25580a1b45bc89df3f201dfff5cf04",
    "sha256" + debug_suffix: "211b67487f4a2c09cea7a4bc7b2e22909128a3c20ff470501a00ce1f078da9be",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "412c0ebb98b9d250105674797185e74e76639a63e997876ea1f871b80d9aea2f",
    "sha256" + debug_suffix: "b0e15e550f46d419ea5230e038142677a57ebac885b270e070046f63be9a588c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "a0bc96c358a521e61e8593247d717e697fbe98bb3831d5a9feac7b5d72a0f35a",
    "sha256" + debug_suffix: "7ca28a5572cfdb26f6cc0fca9fff10bdf9a6d1d5363b4a2d67dac74ac560c87c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "504654cfe2e8801c32e78ef7f5e43ba181743052d40a161de1cf0c767a9e320e",
    "sha256" + debug_suffix: "03f690c6ae0deaa009d0a034ad53d7b16608bf525d83330fbed1c91e6d737210",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "1004268917f92fa1726188ea344a8432b0c8e0e70024bc717cd88dbabbac87f2",
    "sha256" + debug_suffix: "d0d159d6bd9b1a585c0cc937fab09873a50cba0bc03ed663179da84024045730",
  ],
  "kernels_optimized": [
    "sha256": "ac62c511d7759cf92a89beb22961b70cef0532173465f1b3ffd57e519ad30b76",
    "sha256" + debug_suffix: "c64027953751cf7592162c3b12d008b64f0460dcd238924ec725be13309eec22",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9f5d3f945e7ed6f87dba65bdbebef2831dfb9575450303d7bbae036fd419ea1e",
    "sha256" + debug_suffix: "dd5ccc41374bda35102987481a37ff76f063de5b042bf19b93750abb544e3290",
  ],
  "kernels_torchao": [
    "sha256": "81825d00c645c636a5ceb53faa711a1267de338df2fcf83e576b2e9c5267ed30",
    "sha256" + debug_suffix: "d0e3f5142ec7a982c9452b5a388b926592d1464d4a5b0245e670e1a1043e9dc5",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "8e01872e21434abda3b0740fb6ef8b92fcc87d50afe659df1ab85052e81739cf",
    "sha256" + debug_suffix: "081dc083a1a5c343e76a5ea454db6efce3bc88ac46a81527943d93a5f6e9e1ff",
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
