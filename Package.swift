// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260408"
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
    "sha256": "ffad1b3166ddcdee572c68f5808304b2d8321400af15fd4a50477ad359c8c04e",
    "sha256" + debug_suffix: "4947abe5c2776d6bf0ca17884126a1532af1c53a188bbb8ac81c65ec5731a594",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "c16223529622e6812157ee4319d38178289c1fb993abc916a34e874a5ed4222d",
    "sha256" + debug_suffix: "8b051c49b2b235875ec547141422119774b9f8628f958060be5ca01d7fbd1004",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "826c4c7e2966065a21d9bf6df150c29e5f5d955d6492d73df43c9f1e2c01cf72",
    "sha256" + debug_suffix: "1861d32c899691dd62f36a064776167af88278fa4d3b42afe74c27116027282a",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "a08844099cad6275aa912c9fb8d882c22d0cb6a1744bd68209047040c711862c",
    "sha256" + debug_suffix: "fe716590943cc298369fdfc483cf06af96ce46d847334f34a69599dfeb62c71d",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "94f977beedb622743ec2b60eb3315353ad4e18247c40e89912f17402e71791e7",
    "sha256" + debug_suffix: "3f5a4e057e6315e5952e6598e8ad328b2a78714f014938ccbc28bda30b19c5b6",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "d8de40832da0499e5990f5954c551b7f40f2544a5c2870c9a201916f69e2dae2",
    "sha256" + debug_suffix: "5dec6f48c4a54075da61fc9d660beae6e6301b0172527ef40791e34cda06f7d7",
  ],
  "kernels_optimized": [
    "sha256": "35db3cc2df4943ed569a57d274ac13634e54773a82e3af92c6a52a528b541630",
    "sha256" + debug_suffix: "73016e4b03c554dcfa693bdbb9eb7d1aed014e42d5d57d521540579d1b4cf4fe",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a54cb1511eeab654c677ceac15d47040af1c90c323b372756fdba8ececa7acd0",
    "sha256" + debug_suffix: "bc87171b134e35d2fa330394738c0216cd42bfaf75c01b8a183e20c11b0f4947",
  ],
  "kernels_torchao": [
    "sha256": "dd1bf8ae1dab3dfb2f8568babe0d9337920323760737cad47d92232e7c9a3fcd",
    "sha256" + debug_suffix: "01251ab05849d99cd3959f93bd8da7ce87ab1ce48481177566830332b6e192d1",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "91cf64d1eb276b6d7eadfb46c0701c1f329bc51081972356657dac19bd3e6a10",
    "sha256" + debug_suffix: "44c1c708257a1b07e52d83e170c051d10e3e7bba75e1a88efa2dad4cc90e4b22",
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
