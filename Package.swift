// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260510"
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
    "sha256": "141ea588e8eaad325124b1e27f673ee665495660127e0c02070d2a9adf6df345",
    "sha256" + debug_suffix: "1ce33253e25e4b13228285095f371f24ddae1d69b5e9b52e58a3f7ae3e85d820",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3320adef89ae2d20823726741b7fc448f2c6ea24806d368c9d425bcb4d994dc9",
    "sha256" + debug_suffix: "6d1f87b0ce2db9ebc0fa8a817469987e5a9578fdfac69145e5b62d4d98a3191b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0cf9dbf11c9789749f7cc47ca4811fd87f10551cefe5ecd1ed3fe181c11a15ff",
    "sha256" + debug_suffix: "292ac0498f677a58baf6112ed250238c1ae64ce51d7e4d2ec63be2900bcc7c32",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "62acecb4c79e6b4cda81d16d47c60e1c6fdeba2b4dc00eee7e3de313df27a59c",
    "sha256" + debug_suffix: "9031e423f8db64e257c010df4c5acbf04ccb0833b2fc2351de1c1eba135e5666",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "1633b8ec9612dd002348b6ee36dc4d4d49d31df4ce646084c5e4deefe5ec2b37",
    "sha256" + debug_suffix: "743176f778f1116a2859cbc199e4520757430d4615055189aed716415ae6ef25",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "43b79c18772917266a7a09e3dfc11863fecbb344aeb08bf834f1f29edab6027d",
    "sha256" + debug_suffix: "8e78bf833d436723eb45d4ba16032a655a522b38cbd005afdfa8c287d8d47057",
  ],
  "kernels_optimized": [
    "sha256": "a9d914445dd2f7343b4e26c55df3e854aac44402c55cd2e68e366e35081a3621",
    "sha256" + debug_suffix: "e42f44b0e124bc48cebc2d1982401d6c62032209b0c17878a3ccdbadcbc43e81",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "70fed00cf592bf9964764ae32811cb18f272d28ad78a30646b7c6975f92004d6",
    "sha256" + debug_suffix: "aa5ffe1747df03fa39a04c024340012568c2747286d989482a089b9c24d803fb",
  ],
  "kernels_torchao": [
    "sha256": "00bab7f94c3c6ebfb563b812df0db040c9856082973b1ce42f1f3358fe5e9bf1",
    "sha256" + debug_suffix: "3783a8629c59bd740401db3b417c11f5df02e8406466b8dc1ef94395e011ab80",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "718d0bcc9376f7c6a5c965286e929ccef9aee16c63c98ae62952b92ef2462d54",
    "sha256" + debug_suffix: "20abc7b182e83d66faa7e71691ce2989bc353aebe53d163effb1e99ec912bf3b",
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
