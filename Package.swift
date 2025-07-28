// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250728"
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
    "sha256": "ba594cbca52c64457e2f28861c7f78afe81ef63ca2e82743c4ce00404aeb488e",
    "sha256" + debug_suffix: "04ce76a414e912838e2f3967f9e1e1e0ca1c9e3f02bac0c7e0953093d76ed1ee",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "69957cde2c30e85c8adb22ac490d5573b52aa45a2d3168f554726b190d95b2df",
    "sha256" + debug_suffix: "6f05331f86809c220f07d453450434b0e8bf6a33fa632386cf09e6464a47b23f",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "648d57827118c4c4061bb6221ea7dcac2c82ccdb1387939e5e732411f22371ef",
    "sha256" + debug_suffix: "e8a44b92e71395969744400735d022c290e0631e748417438ee0e4b89cb7c28e",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0a7a06acb7778f84c45d96cc2eb34478e4f539695b9c1708254db7d4c23d314b",
    "sha256" + debug_suffix: "561f6a6a65a400745ef5e91c3b25f7f43ee91e645b013b04f247324aceee51f0",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "baae4cdc45874715e3c6bb0b6df0f9683d460a94e2aee5959846e8a8df2acc95",
    "sha256" + debug_suffix: "04243e23d9ea6a90ed6f6a54b9ad67f550549a19c9aa152bf007780c7c5065bb",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "9707393c29112b77444693f0bd42e4fbc2712c44dbb07e277730c41243ac23cf",
    "sha256" + debug_suffix: "a45d350d181e0533f96ebfb6dea48bda320ee3fa5d943ba133efe2d914af508e",
  ],
  "kernels_optimized": [
    "sha256": "a2f8f5b17535a3348d7c4234ffb98cf4779406b5ed14c3c61fa4959e850ac28c",
    "sha256" + debug_suffix: "79979d54d2cb0697a9d126507d5ccafa7dbaa5a7f8be3bf75c7477b7eed466ee",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9382a160f1938ddf40dfea3c2cebb7540d381c9d2abce530a1184738fbc79394",
    "sha256" + debug_suffix: "b080e33a89ebf0d8333007c5dbf178979a9b7968b33f47bace359472cffc9a64",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "75ef677654dda47fef0f055f201c900c3b81bbc977326107659293f8af4894c8",
    "sha256" + debug_suffix: "88c36cdfaaa6948cad12e927af67946e872200214b6e4b16c387530e8179cdf0",
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
