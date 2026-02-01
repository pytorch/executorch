// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260201"
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
    "sha256": "c25d0c8c4f3561165a674feaf7fc92e0fa32e49a0c9b10b871c9e17c04462a58",
    "sha256" + debug_suffix: "a91f4d0909e7f95f0d1c691fa33257abe21afae2f666e3f9b73335516841ac8d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "19eb2c38475331407a5b5895c4536afcf9f22ecc268bbed49e34f84fb0a270cb",
    "sha256" + debug_suffix: "6b3a93e99dbfbca29ad01a6bf387dc1458afea2774c4617d117a2870ca0f5f2b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0af08d2493f957311efac08f45c543cacd88dc157bfd2d03104034f3eef7952a",
    "sha256" + debug_suffix: "fb5decc66d1312019e8c4ba6818b4970943604af6b3a52340bcd632dd655d468",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f1df71eabcca6079a5588f780e8c463d50c9fd5f1a2aea598c4d98d5dbd49a7c",
    "sha256" + debug_suffix: "b3c0ea63477eab24dcec8a03b00b04edb216f123ab2520853908a7bce033aae1",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "2b2ff7729b7dd355e13bd91997f154b139a046b95b898b2e4115859790af7c8d",
    "sha256" + debug_suffix: "7f095dbb4dd484f24f5b2ded57370fb0c1e495281114a3eee95f112760821b6e",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8330e02fb413636acf92040d862d1eacf5c6395048c28f15c13cf4719b546225",
    "sha256" + debug_suffix: "54db7af39e413ecdb475dbb916903c10c3318c7941327f633905e22e7f29b6e0",
  ],
  "kernels_optimized": [
    "sha256": "53f037e01b59d45d32374cc9476551fb391be0cae89a71c1beb2ced473312fb9",
    "sha256" + debug_suffix: "818709997d2773431ab492339545f51f3a8e7415e9e55d07d697a0627579f929",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "05a61017b437bc3e3b691753b3b906cd1b80fae4f1a5172ef5d762a0a581c90a",
    "sha256" + debug_suffix: "5b8e0cd7ead4e57c861b777a9db8a66f74af1abe0fd38014ee93bd27e223a9af",
  ],
  "kernels_torchao": [
    "sha256": "69bffb587f72a2dbdf0d8f5313d02d4ca5bd5774059c334f911c7772040d523a",
    "sha256" + debug_suffix: "cd9a8d8218b50395ee6976058e2cdb23a0b87eba0371e9463f041fbcd58ea414",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "0d88947cb60c093cc155384ea6b3e3f5133fad2be7af8cde7cc5f1a612af9033",
    "sha256" + debug_suffix: "835a280400ac48b9f3f4e0411faf2940bf02a21047994456f54cc92548f4e9af",
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
