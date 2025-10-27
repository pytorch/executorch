// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251027"
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
    "sha256": "167520497c66c5b0b8d140af855ec5d1775439939dbb5df985a30b6252100417",
    "sha256" + debug_suffix: "a3b6e4e9c127615600a974aa29a170cee9fd314c072c3403179faf507e0f908c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "862671acc2d7d33912c2830285ee5ee25e715a83393bba36be5cb356910a8aec",
    "sha256" + debug_suffix: "c80fdcc223ae467e27b875f531573785afe285e203989927c017a5a094eb9086",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "efde28e2b44f691f94070a07ea3baa79a34aff0f50721371645b131e534849c8",
    "sha256" + debug_suffix: "bec76b1d59a721f8a38dba6711396e309494dfa30c079f80af247d1ada5e2f72",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "3c0a901eb51114dcd4a93573c7ecb730e70bde74b271820a4543da69fc28bbdc",
    "sha256" + debug_suffix: "c633365f92590971e184c2371439f3cec2e0a9d65b5e840ffd7203965c65c5e1",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "d0d9246e1bc8c46d2af8eafb6a645c3a16cb03b14dea17d4ac5eae005fe952bc",
    "sha256" + debug_suffix: "6d03284773ff0f4fadd45f734cf52406b6038139bb95b1879b54cac969fae08c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f3f34f1b773a754f08687ad9369452a1f00a96164ad8f824154921654bb7ee9c",
    "sha256" + debug_suffix: "3956973121210f3d10917515f73209bd1f64872a5f464e14645bb392ed114aef",
  ],
  "kernels_optimized": [
    "sha256": "c15fabb6f08b2d39f8ee6e3416335714c474edff21bed93047f6422874bf3b1d",
    "sha256" + debug_suffix: "b17850737246eb98867dc96a405e1d07022f60f20998dd50605dbf0db98c4011",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a5049f8e0355c0127bb141c44e4c7ea783500b1ff242447b11e5c6d19b6ee84d",
    "sha256" + debug_suffix: "782501eac5b716c4079038884539cb6cb72e59e72dc0c1e8004d3fd675148f67",
  ],
  "kernels_torchao": [
    "sha256": "03e3b572ff167de99ace383a0619c1c18b13c774065b6a7ea5d7bfcfc48cbb23",
    "sha256" + debug_suffix: "64ed153f3e508a6407e96bd1a1eb619a1fa9335b3928f1c3acb423ffed0bdeb0",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "72f423d4c2ce838b991cce2a8256daebf8bc97e2f4d6c0fc8d23c11d979027f6",
    "sha256" + debug_suffix: "85eaa3c7a6aba69ac698fad8e2e9ea0bc5a8fe460ef2c58a5dac3bba9a9a0326",
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
