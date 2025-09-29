// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250929"
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
    "sha256": "e0f4ab2a98c7a58a55e1283e0d4e4227bdcb1add3d29b755a9a252f59f1a2fe4",
    "sha256" + debug_suffix: "af34485efde3d9f3866006b736d0f29bdb5621bbe9b4b2b4a87ad99ecee64812",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2a151b09fa20f3fd63b01581e4aa8ce287c3e7bb3d32f6643ba14daea9d5f650",
    "sha256" + debug_suffix: "4c409d3166da09e5c6f71fe5681e45fab28005e0c1870f0204c7f970da953037",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a4e750051d2da8a752d1390caf71ec1eef8cd27ab15c6de791f6d7853695c26a",
    "sha256" + debug_suffix: "b7ded6efcd0c7eb9e1809751ad1fd657fd3418d18eb53dc21dea9c9ce1583983",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9df1280ecd5708fcd9cd4f307931dbd5f15cca6d9b720624eed36ebf52220a35",
    "sha256" + debug_suffix: "07e7ef9539af4dd4231dc45d750ce6c54e38baca9c421988ed6d7e06160fb37c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "f318fff964246af4c2119a394ab8f65a2edde9b7eecfb618bd4c262e28c12f1c",
    "sha256" + debug_suffix: "de259753f4f68111ceb6dbef8449c8e095a767c4299c5377bd76c6b6019a2de2",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "001b0b0ea25b35270d34be1360b3133e22c46d8d1e8c268341e1fca528b4b9da",
    "sha256" + debug_suffix: "76f0a7888311eb43f250fa363312adc8e045f559c90d3f2c868f985b66db60b3",
  ],
  "kernels_optimized": [
    "sha256": "ad41068d838b7b4b5fd0145a996597fc52f70c6f913ec92a109de2a9e4373142",
    "sha256" + debug_suffix: "d341042fc861c5677b68d811e9dc456608709e856d4fc3e97de47b66fc19a888",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9312f275895b6a973e3bb10df52b29ade57e38392b8a3f07013f892d6bd51dc4",
    "sha256" + debug_suffix: "4f49579063b5d3871cfdf688b5211ee3f01167ffeb2a58276e7464ed9428571a",
  ],
  "kernels_torchao": [
    "sha256": "394e6e68bbee92c41e4da7279dba8e46af490ba3cfde20fe77cb2a89f0fca57f",
    "sha256" + debug_suffix: "cf0880f76cbc6230f303feb28262799a677fd9f3e4a4d6429d2e0b38f4894ea8",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "4efd7110ce77278a63100d8e80007578288f7ac1cac3933813c14a52b0141014",
    "sha256" + debug_suffix: "9e6d4d19666b766f4b58940f00d614e26d82fd4dfec023f8385d7938d05668bf",
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
