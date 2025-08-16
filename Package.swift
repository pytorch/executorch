// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250816"
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
    "sha256": "09dbe7fb163effa3d7fa3d775d2d5c1fcd823d4179adedb450304ed8929cba95",
    "sha256" + debug_suffix: "ba7e833fc0509ad67ec1d82fc933bbcd0ed435cf0bf22e8f984c50ea68dd353e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "955e2d77ee9858cccb92b147d51aa4d531e6431514466af1ec16570714623970",
    "sha256" + debug_suffix: "07fe99cd157ce05b42a7bdc3a259e419cd504cb1d30a4e29645b02cfe8e35b60",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f899eea358a14818a1be89bdf2c20f02dfea3269f4527f2e4e270af1f99d12a8",
    "sha256" + debug_suffix: "460af651cb59bb7ea821457dab6fb74a304b366a8c6c355a5c92f5ed123d9daf",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "902064a89159630e7e8019f783c4009ff2b629d8a24601e73826c434602eb701",
    "sha256" + debug_suffix: "9c8c9ffd2fe9a79503b9708f0dd5726b233572ae4e647a2b4053a4d6c31701b4",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "f5b60cfc30efc27eb916f3cac00ac0062e564b251214e7f95fff51fc525e8b24",
    "sha256" + debug_suffix: "811435da92f84534c5b088b4e1a87a67e729e1d683a6695baea1b72a9a8b5ea7",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "a3aeff5222100961bc7b8541399594f7ad7eeb59724304359eb0df089b2ee275",
    "sha256" + debug_suffix: "6e1801d8675add313d15a047ac94e105700950d61850d3d2be9f6fcf71fd6d70",
  ],
  "kernels_optimized": [
    "sha256": "ab0662664f4f2ab7f9e7c9f67078d681060438a3ce7313a12b0da76c072122d0",
    "sha256" + debug_suffix: "0a4e6ae9a42a9aecfd3dc1611eff1f20721fbc2f419a39050efc24c730bbe9d4",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "2c56c1c87620e19acf076357463b9162a5d0ca3082e08810f29c62a645f701f6",
    "sha256" + debug_suffix: "fbd99207d9fa44f2cf4a5eb2950424a536b2d510afd715fb2f850b33e32e324b",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "3e704d8e5a867db3e832870d5f904b0cf0c99ea29b9d43cc84df0382696cb9f0",
    "sha256" + debug_suffix: "1a7d29f11c7cf37333a45ecdb39576d370ad42ce18b8aaa822d081c72ca1baff",
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
