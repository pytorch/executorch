// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260720"
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
    "sha256": "cede52dca8f7dd399ddb7890f6d2e2bdd7aa3fd7b9c9fdddcacd729dd632a3d2",
    "sha256" + debug_suffix: "3611a5935522eff84424f1e0c08875b49c9fa15d8d7be4b640fdbe45e110e68e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d4bf03f8a65b17b7d3f515a65e314f79159723dfe23048ce19c1ada5ea361feb",
    "sha256" + debug_suffix: "62359f28f6b3973fc9203566ef3343680a25b1024241ef46e7a2c24971af44f4",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "06fa350e563293499fb2e33da7f9b5b748fc71eca9a8f8ae09e3cdaf7169a78c",
    "sha256" + debug_suffix: "f20e10e3759e17758db4ff163e1f9f22472775be2e4a8120315f59a8826e1e51",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "09839a675ac82cf76455257c0cd87c58a145dcc3d937c864c150fdf3e37bf644",
    "sha256" + debug_suffix: "630b3c5e4a69eb4c2e90e61e188fe501aace53cfc52ba7dd0a0cea7717952f47",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "bc4cc45197e2be25d456a9f1dfc4f6edfc8ba934636a9b9cb4cb00d2491a0034",
    "sha256" + debug_suffix: "b296f19abb8606564c2d15b8829a319fd2dd7ac027bd0bc1bb46f1a30160ef39",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "957afc30de9d49c2cb9742debe9eb56e84b229d5d26841bcd52751e49d04ec78",
    "sha256" + debug_suffix: "f813f95729524a9d528d0d94b764d2a1f8220149ffa1854bb545bc22ae5c152f",
  ],
  "kernels_optimized": [
    "sha256": "05279e89a48f6e405a26ee9e1966459120dc547fab167b70db03842e6fce5db6",
    "sha256" + debug_suffix: "2f8e21118f1ed750dfc3212e9150caf3f0d1fcc79f691898624dd0ef1781d407",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a0c250f57a56e72c48fee64800bb6d22fc530cd6bafb1727fe04f9687637df22",
    "sha256" + debug_suffix: "e99a53c81284b7c5038c2a13bf688fc2a5aeac8e4679741a3c036abe254dab0c",
  ],
  "kernels_torchao": [
    "sha256": "b68a19f032e1c064e6f7684cdf05f334ed11a4a6a448e600e43c9d9bda498341",
    "sha256" + debug_suffix: "749b6de5e47f1b4e45908f8751e778c5f56eec0386a0dd04faea3d9ec03aec8a",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "3dd1ae07ca477d1ac141ab38481ea36541adf022f77d2bc9c4e5fbe12842a4a7",
    "sha256" + debug_suffix: "3904c299011a896c8a2d59183360194892fcfa0f5f83a2bd7abee19cc78227b6",
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
