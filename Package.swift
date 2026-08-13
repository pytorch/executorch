// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260813"
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
    "sha256": "0dc2463aa3a9e75e4525df049bd04e3ff3c4a61576cee4f609391e9e85b36f31",
    "sha256" + debug_suffix: "866c18c1e74a39697abc6696b93eb00f27845921a4224ec893d2df0054a14495",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7f414251d2f7317f565c7fe9b133ac250da7850f83cdba4e3b3692dc3de21701",
    "sha256" + debug_suffix: "2938dc85b5a627dcb3d2f7859be5d5458f688c059572a462407d4a5c2facd23a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1fabc62b75b8267698c6e38cc5215052dfedf297c954e62898f1e53e92c670f8",
    "sha256" + debug_suffix: "9d690285e27a094c9598b94a5e608c9bdbdaf12ceae638e9bee4f227d75bde72",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "30b62e14aa3cf46c89f5b4c39dfc4f01c55257ddd131f19347629267b708eefa",
    "sha256" + debug_suffix: "ea2e69544efcdca6c0d3309c4017f9314d6895a11d95577ad7219ba3e6713748",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e3b965f8f68d0ee57bcf32b6a10cdb923b308de62c20dda69992c6c145e07f5c",
    "sha256" + debug_suffix: "842a9ec650403de0073d26b98a2ffdad3a1af89205b70c73b1f8cfd3182d3ff6",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "99ab8e55009c326724d6dea24c94145de1dfb2835b17d8c62944f938c7573c99",
    "sha256" + debug_suffix: "a5b6bb7e3fec7c889dfcc49e8bb56f64a343108eba728d79d4eb086598085417",
  ],
  "kernels_optimized": [
    "sha256": "d8447553836ec7c16d1d49c01e298f04b0149adf3bec65e17d25b4834d9899b6",
    "sha256" + debug_suffix: "536abf661bcc451ab39859845af03f2676cd57499d9db16ac0256b9d98691bd6",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "8187ddc7fb3fffba162bb8ab1a20d3afec06a81e37c18e6a88756970d53c7dda",
    "sha256" + debug_suffix: "0b8e5478d2ef770a81c193d8891af604dc5b1970e4eb7c6d6fc8dde9e1d3f201",
  ],
  "kernels_torchao": [
    "sha256": "bafb68908c4201883259172f21e4360897b11d9a9ce49d5d7651b8fa9d9dbf0a",
    "sha256" + debug_suffix: "3cd5f3b6f35d6d88c65639c736d6893eeef2792492ce7124a140b0ccdd4c4dfa",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e2e9abfa1205be7294ed99962a41f85ad4b5194f58f611ba4ca7eb1803213511",
    "sha256" + debug_suffix: "0f6b4002cfad828d87de1ac5f7f98b9c2cb52560f306791050f746a2b8a87704",
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
