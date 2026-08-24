// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260824"
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
    "sha256": "1eaabb124d2b81381f301c09c6e3963da3626890d17965b9696b094a7e9179a1",
    "sha256" + debug_suffix: "0f2d0414e52517ac469ea1152ff4cd34b94dd0c9a62c99d83eafad12153257bb",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "02ece07f7ea10c889422fd63c7844cb8edc65e104bc42a610999f45ca120ac3e",
    "sha256" + debug_suffix: "67d70fea206ee284a82df0a20d6345751dd9bc1552e4335d0ac30a9b24914031",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b5b46cd4f7374896ad990d1a5e208621f3236c1c31966f6903fce922fc718f74",
    "sha256" + debug_suffix: "fa8ef8b5291ca04ed27d4361bffd5d4b91e38923bb6b8c1c4b384300c951708d",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "27622dea8970a2cd81d85cf718b62f81710cb0dfe008774c81e6c9b551c7f558",
    "sha256" + debug_suffix: "ae7b93e8bf2ebdc47d17075d79da650af8fc3cec88de1b4176fd47197aeed76d",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "f1f08022179c79233cd843a7eb1ae5d5d0e5ba52c219c57c6d2d39d57cb0f7d6",
    "sha256" + debug_suffix: "84d38a685e512066e50d9e55ea453a2a24f78c40cc32fc94295412912286c60f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "b5a2b9df6c192016932ee71806a22034b9da2a6c6537ad4f1db331a440363b98",
    "sha256" + debug_suffix: "c3102fa99c6cf64f2c346aa739d0ca6a93be9b4a57d1e670eb692940cd3758f5",
  ],
  "kernels_optimized": [
    "sha256": "bf7fe57f63294bc0702ec9764e96b302cf44c5020bab5f31e3d26df2fd9b6630",
    "sha256" + debug_suffix: "42c27521d06eae0367960b112f70153b5ce964cfe85dfbfa12ea5f43975fac2d",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "76afbe3247f895e4142111d5a1c35560fd52b20b46e1400060e65720f709e276",
    "sha256" + debug_suffix: "457d44188152d9d76bd627506417b30dc9e96028b4f8aff56d7010ed13c12ea7",
  ],
  "kernels_torchao": [
    "sha256": "d93b9751ca2f4ce923903a705c74257a1c3dc404137949335c0d8becf600f977",
    "sha256" + debug_suffix: "8fed8eb0a8964227bcee746abc49dd4b4bae363edd3749e4bebc85687363e878",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "05f306e1ee332e3fbaa09a19bf8b1becae15bb0a761ba1ba7ef4edf5fc6661ef",
    "sha256" + debug_suffix: "088ed817d0c7222f1a46b6381f6550e56e2905c7f5a02863bffefbc807c057a3",
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
