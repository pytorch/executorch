// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260708"
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
    "sha256": "b5a399c883b980f443a5dc62c790802fac9f7c83b2c30b03d009dd7654be1c85",
    "sha256" + debug_suffix: "99f6bd370ab73a3512769858a9ade6df6a4ab436c844fb84b1310f1407e59a16",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "eb6cb7fefedd3076dde6aed019a12e70c7369905f7c55ad4d89ebd664154f778",
    "sha256" + debug_suffix: "16d20c3efc56e8b4d34fb4992f7243092346769122fd99d69746bd0478feb96e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c83542c0572a88a0fef06019e2d8830d62ec0c204f62b4aed1c4688fccdf63be",
    "sha256" + debug_suffix: "902581f7f253fceadaea5b09bcac90e572c1eb7ad9807fe9387c74ecef3307a5",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d8ce4b979dc8601b886768e4e5e1de0b5982ab67be39ec93daf79be9ebcfbae3",
    "sha256" + debug_suffix: "0bd5e3c606cf20107aadba860a374d07de28f9945f8d5c53da18ea1d548b92f8",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "cc916e949d9dda57d049252fd6f82fdd9d87283bd96c0cdecfb6ae6937cec783",
    "sha256" + debug_suffix: "387268d4c1838b738b3f92cb7bdd36040791f428271c8dbe0680912d5242edab",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "39923636112805b303596055031d6b44d0d3cbb3784cd6f10e70da664c2becea",
    "sha256" + debug_suffix: "69644c7a8b3fc637d562558b9f6d50227dacf884d064891aa4a260680b1e03d9",
  ],
  "kernels_optimized": [
    "sha256": "898eef4503f44ea77eb0b2123a7a9342ce5afde6e09b68de99cb0ff0e0029a62",
    "sha256" + debug_suffix: "d2f5f0f937a179c0548875360242ef601aa5df3922fd097d3e9c84cd67a08271",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "bc635ee92c298a648bcf18c197f3ee743c835a5c40e6ff95f6dd9b857b677f2e",
    "sha256" + debug_suffix: "fc8a911edb727c0242ec705c789b91f399527cc532b189579e56df7b60b31c9d",
  ],
  "kernels_torchao": [
    "sha256": "d920cae4e395ba5171b5ee1536d711a8b6d3836fb24c4ca2c2549d0377dc13c7",
    "sha256" + debug_suffix: "29114efa0077ebd26bd19adcdc9e73be596f0e041ca6bf5133a35d4a9b69f51d",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "006e73edcad1db92d9172e7a75bb88d62cfae06a091994df6a28a09b5603be0a",
    "sha256" + debug_suffix: "01eb4873be9813d3ef8296e2cee2bbacfff8d86f67d175b8d2c3010d630b98e2",
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
