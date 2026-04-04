// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260404"
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
    "sha256": "4b7ef2c9169d928a6543cd1a8928fd9c9bf505ab35d852c3653a1fad037f905d",
    "sha256" + debug_suffix: "cc367b39e934801a68c1650031dd9954dc43fa0748deed921bed84da6a7880da",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "981bf35e978e8677aadfd970db09650486a83ec85fad5f892e90d11722e81b6d",
    "sha256" + debug_suffix: "5fa628cad6e47d4add53aa50761012f874e868712fe702cc2c8cab2b884d099c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "56d3442832783110474de473adc179b9a2cdbc55fb6fcd07af9be953fced3207",
    "sha256" + debug_suffix: "f44f8b66ca850b25296d2c87acf19f9962947665367f6c75d373e38103d7a40c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "68b785582444185d9d487e7cf92fce1a63280a2fb90c51302af07174c2ee89a5",
    "sha256" + debug_suffix: "af942c664a1776ffd0bb5716b5dc4e774f60abb22fe2be85f56794bd011e3786",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "f61027e1f550288794e75bbd31e99c93c3f73efdab62707451a84eaeaae74ff6",
    "sha256" + debug_suffix: "77556f1ea1195ddce02e5c7f330b8724a77539cd50da69528cf7336619b61aee",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "82637c748a52b907d152b61c6f1f41e6368f60815b2ae9bbb32ed49810ace72a",
    "sha256" + debug_suffix: "5adbd13b24d10c72be8a04ae8c55ad6a7e78086ec3d4c8004c66c8e9d6465f9c",
  ],
  "kernels_optimized": [
    "sha256": "1fc44626fb4d1dcfa9e506eb0eee4035e3e57dc678c98f5a27ba56011aaef3d3",
    "sha256" + debug_suffix: "238e295613ce7ff3364452f97ddf459eaf3ce9f7313afb5a0f703280a30f39cf",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "573078b3c6c351dec73e7bc52dc850a842532ece0e8ae7a570d3db6183d9d7d6",
    "sha256" + debug_suffix: "230bb0323f6abc565570907d6cdfbeea931faedf71bd9682a5095022571f6162",
  ],
  "kernels_torchao": [
    "sha256": "16d4baa04f8428f13e1002c7aa883c837a4c28c0c301100453bf08c85bd622cd",
    "sha256" + debug_suffix: "4c188aacb7524bcdcbca999b45eeee7342de7214cce7fca2f27f758836d22bda",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "cf563f0e4855a92a9defcf831318273c55db9484f554e9cf5528a6ffe2c04eee",
    "sha256" + debug_suffix: "40f11bcb4346192f4bf200d9b8fab030af4a6dfb2c24180e6ff312aaf55ee6b6",
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
