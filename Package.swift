// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260713"
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
    "sha256": "d510327a5f23e9e89cd1197c687cbea9b14a60dea4528ee0b73094bbaacd82e9",
    "sha256" + debug_suffix: "4d82b455be38563e86174ad4086f48e8977334a7af5498bda996b650d33b4df1",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7c58cee9b902d69a53049453f9aa2cc6e047cc3697bf86357ae4aad3f58c4d7b",
    "sha256" + debug_suffix: "53cd717fd78a284864a06969d70e63ab75247fd406be4029fdb95dfcb22c2762",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "eb6ba61d92be20d7bc30d14efe77b014da94e768d3486c2ce543ca1a8ce920e8",
    "sha256" + debug_suffix: "d5beb01b2f1a142c0f7fb5bbc7b86848b3310c824f904459f697eab7c665aadb",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f4bbce1f59fc891a60a125b66a5fff16dba8595cd2b68955f8f31a37020c1f88",
    "sha256" + debug_suffix: "63dbcc15be4219ab3d1b7b1e230a0444e111bf5973e85161648bdf02ab118fd9",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "56de482cb5e9af7d1414e1cf335e01ab939183b88cf02570a4a8955660e96f32",
    "sha256" + debug_suffix: "1802cf3ecfc79c9f6ac1152b88e2c939978e55d2c555cd2bab091442499df5d4",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "93f462984cfbdb9eec34ca1f6b5d31f3f75bd61d6a20e056ca8ad9aab7124db3",
    "sha256" + debug_suffix: "773bb0f925ee8b5a8982fda46e341beea79e12dc8c022045c4feeb7cf3f15dd6",
  ],
  "kernels_optimized": [
    "sha256": "e68dd399e4a48dd173bbba8f31351b520e96a557329e0e321950f2a121c1a50d",
    "sha256" + debug_suffix: "4b017d29a21162c79c48a61343e901232a4b5d9de54b60261d340fe326119394",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "26b42793e3a26b766f349c09ee8751823b3632116a63533fa018b891e7053f01",
    "sha256" + debug_suffix: "6022b5da5ca9c1f186faf4d03a6cc65c2052b8e5078b02482b98fdf8e462eaaa",
  ],
  "kernels_torchao": [
    "sha256": "4b33452a5f889d99219b2efc7928f6ff5d1a15ea0d842bb1218090da820260a3",
    "sha256" + debug_suffix: "71debfc972c292c555ea2413af4ff43b1a38d843b0a3bd193331925de929f4e0",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "f4748753a54ee286c945d5c61ad5b4920b1dafdbae4335c55e3faeffebf2ddf8",
    "sha256" + debug_suffix: "c7236aefc89eda050e01c6dc6a53b46202b3415fbf59a953004b83c8e59bcb80",
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
