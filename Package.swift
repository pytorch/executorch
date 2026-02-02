// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260202"
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
    "sha256": "8e401a14b68cfe00fd2239c78e509d46e44673385154cf4526aebda4cda62b13",
    "sha256" + debug_suffix: "9c2096547398a8237ab5246473da5ce0bbcedf4b7495f7eeef90748623a0b935",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "903801730dac4a7960934a42053c24d21622bf9c8f308db13706c7a7f15fd02f",
    "sha256" + debug_suffix: "356ddd30dbc0e163a66470b4c30ed4354a09bb4338f7459066a41d3d24a19d08",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "81e95a0b9e859bccba0bdc833a899a0899b059fe7a8fd0592ab7ca005a2dfd0b",
    "sha256" + debug_suffix: "507c7e347afcd7e4d1f834ba8ccc528bf4bfeb5b1ca40200dcce2efdf526c77a",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "56b69f8649f580d1cc57831e5250370eff831a4856efbf1281629d6291718915",
    "sha256" + debug_suffix: "5250b748e5895bb882c87a4a886c58ebbdff1f86ca4e04944c0bc1fb442a483c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e9b2c8a06879c73110e06b6df5b6b952a5b14ab83e05046d546a08a5659caa93",
    "sha256" + debug_suffix: "25b50862e033f631499c06f540d6d37ef4600f91358f1f4645bf5e9515e01b7d",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "79f5ed15323376754e7558df2a884b88dbdad6f7f3bf9bedaa4d4a08425a87fe",
    "sha256" + debug_suffix: "a6a9b9fe438e66ce5473496b69e6a0239e98e74acaeb231ecf7c0c03e698c3bd",
  ],
  "kernels_optimized": [
    "sha256": "7368fc5a0bdc4912097dac1ba8a6c08ca2fe7b4004cfb5e10de653c3d1a17db0",
    "sha256" + debug_suffix: "e157e222f0357de2cea5ca6e0d39437e981a2658e085526c28acd9470f49c928",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c2c6d729f3d3f516a0af4d2a3c2e5e716fb213976b9470ff3151bf2fa2b3832d",
    "sha256" + debug_suffix: "3189408b156fc2abd55b0ad28b598208438c8e66c06782183df9d46de370f3a7",
  ],
  "kernels_torchao": [
    "sha256": "4baf878249fd4c64cb935e5473b9351ae40a3abdc037b9ff15cba73c911fb592",
    "sha256" + debug_suffix: "3bba48dbf5695614655d2227ca402ff07c2586d2df4210bc04cf812abc721514",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "d2e64e44a915a004d67bd5b143c8d20f444d5984f2f22389f3d1fe5121b0a306",
    "sha256" + debug_suffix: "f55e73dd21f6ad56ee5c261505c95ee8afe39113094f3543bfa3a0edc60fa2ae",
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
