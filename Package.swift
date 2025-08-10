// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250810"
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
    "sha256": "eeca477fc457388eb6ec8046e4b8d743e990b625379509284fab95c82941e39a",
    "sha256" + debug_suffix: "a542655e145ac073ff3e3ecea323d1da44fda0736008661a72cfe52876a782b0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "c83b669bc97639b1ca52984c374d6975e58d53a3b1e365a15e8b78e842f4571a",
    "sha256" + debug_suffix: "cb58123b12f32808c272c0d0a17e6f5e3111363eaedc8ccd7378bb4e3179436d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "dfafbec47ee5611659290b4fbbc678d711c1e2ffd9766f784d726dbe8ac96c95",
    "sha256" + debug_suffix: "d1d891ba321e0529a4e1355745763733a0772306cd925a15c893a02b01aaddc7",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "80a38cd68525e8dfe437823d3cb36eb577f90c1bbe68fcf8a6ba67e847335c46",
    "sha256" + debug_suffix: "138b460ab8c1ac88177f60ca6b46cf8ad3b5fab40deefbf9826dfd5c1d69d385",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "351dc6a0c37b6b3316f0cbdd5d128e0552f5aecca80bfca1f497e4e40be3c95c",
    "sha256" + debug_suffix: "a27cde91ef9743c7641d20fba9e175083b853e3f13eb84eddbc54770f790f814",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "2c944848c460406dd28da91367427d43e49a01953543ce943e61662dab06d251",
    "sha256" + debug_suffix: "2f59e10c549df2a44d0591eb44be4ebc7b916d93c34dd88a3aedf05fbd7ecce9",
  ],
  "kernels_optimized": [
    "sha256": "f0aacf96550588f87f6ec2fe67dda45be1490a521256ed67db9e05807740310d",
    "sha256" + debug_suffix: "a57cda1c058b0dda7c1eed1e89c0007a086423c9d04056a4555d78cc94a3420e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "2938eef92d2bafb86986c9685d2d89d9787267a20c95eba9bd11d7dbb40d1195",
    "sha256" + debug_suffix: "270168bef820773bb84f22cc2f083eea0654f14b22da600e5db2e4a1358d55d0",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "ad13d79248b6e45ee1008ffb01dc8ba44306d5f7de1c4e73bb6044f8f8c85ec9",
    "sha256" + debug_suffix: "136a4d5ff474f75de4fc91fc74e0cc7750d0d6c2d3ee37484172d57c895be67f",
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
