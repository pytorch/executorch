// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250729"
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
    "sha256": "b784e2884a2e8acc0f8a6aaa42e7d3bbc5b63d714dd8dfb070a60761db30506d",
    "sha256" + debug_suffix: "d1ff082f06535f98ce7e699bab22a3678ff37366d0c8bf492681e8b449079a33",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1bf22e5642871810fec1e9916df1d9a18b207c68be5184f9c41b332cc62d0880",
    "sha256" + debug_suffix: "d071c04bdbaf46a6eb78bbfc2ba3bb83d907f1207da7490650d0438738d1a9d7",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "09a584ef9ece39b887f6100abcab3c93d07169016f1cbabceebc66a455efe590",
    "sha256" + debug_suffix: "608dbf7b7682dac785e98e4aeb85db35e6adf4076040c12b7aa03a7bdb32316a",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "6a3ab43cd23a9c3e3fc136d8197086fc4cd416df12f3880ef03873f667d32d15",
    "sha256" + debug_suffix: "1692c3cf0554674f3bf29cd291d5a761a0b14ac89aec4f8e88e13bd7cdc4eeee",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "91114c8c5f9008058236d5aa08c6bec0edcd813f56c01d550d651028c1f01a19",
    "sha256" + debug_suffix: "fcad9f64c5df86a449602dcc53bb10ed5d1860a433d8d450ab317fb12a37bbed",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "af98251508c5a02835bc3294f11827ba04f1856aef58854c4c85e4ddd6b72039",
    "sha256" + debug_suffix: "8345c6e43035ff2d5b9627da4f643daf4f460f14dcee170f76ef9ddfe47a1445",
  ],
  "kernels_optimized": [
    "sha256": "84bbcded186e8b4ea572f0b43fef7e5905df29037a69bf19adee77caf3876aef",
    "sha256" + debug_suffix: "6acfcef45f5fe032b71e070cbc1cbc6ca40bd1e3368f1c75e93cdcf33bbaf81f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "40282e5a1dde0afe8f93d76daa7a9aecee1370c4fc121868ca39faa20f162f11",
    "sha256" + debug_suffix: "2de1048db8f3facd2ffb26c54310ed93e689da437f389687f5cc6cdde35f2422",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "9b6ca8de8e1989aaed68e257969db74e6adc091f1561540e76c93e8d30191505",
    "sha256" + debug_suffix: "c36112cd1d085026880af047ad80c6fab0d944ea2ce320020aac7f1fddb2936a",
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
