// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260810"
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
    "sha256": "a7359e8ddefeeedff5bd5f8f4f28db423a0b9d8455a5a997c101193e823fd42b",
    "sha256" + debug_suffix: "c70bd0047ba74e388e86adb8e4370767fbb8af2d10c7d2dee300a4e2c9e05e09",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6ad232ad0087d4a12132089b70cd4e344ef6e7c90da02c863cf22a4335d6a764",
    "sha256" + debug_suffix: "50808022127e104dbd8aef7b7326bfeef2077581f202a3d607a6b21c3ec11a8c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "57d59d46a1730c0cdabee2217fe23e0351cd29bd1248e20bb40d68da1d00c60e",
    "sha256" + debug_suffix: "1e7ed0628bf15ccef4d9bb06f080aaebbf1f444a458573d94ed6f33bb6a5f561",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "a4d647527a562ac3876793ce7a2a3d9e7aca7813dd089b56a701f9a498a712d6",
    "sha256" + debug_suffix: "62d48f19972504ba26f1b7e67cd1dca471f3ea40e5b667cf371fbbbda596f365",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "674e7bec82f0826d9ba096bf6d04c48ce6c00eefd03d39a77e6704a0d76c2ac7",
    "sha256" + debug_suffix: "17ba60b3710b4c44a0778725cba5513b14a1c3043ce263fe877e8f2c2f0fd3f4",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "3afc50492e2e43a1c24d31ab770047e39240765a478acb5bfa6e1a706302cf5f",
    "sha256" + debug_suffix: "f04088484d6eee09bd82f105e43f48235a5487a34acf98f38fb165b1655cfbb6",
  ],
  "kernels_optimized": [
    "sha256": "d991e4a0f5e22b7470bce30eb48a16d563250a4a720c49a5e7d851538b148f03",
    "sha256" + debug_suffix: "cae718885fdb02e6429ab5503f5c0910536a3f2f4cf4f7b0f76506718b112bb3",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "440ec28173902d0cddcca87215e940b3a94b365bbf7c1ea654d6c2966a8f69e4",
    "sha256" + debug_suffix: "30583916eaeecd1cc0205922624a4629c6def12f87865ac0e6e83c2ea4b3e84f",
  ],
  "kernels_torchao": [
    "sha256": "9277ca3f28431949f0cd1fbad2db138a803ca2302e1859db23d8e18df6f72804",
    "sha256" + debug_suffix: "8c3125a4af2c3b80dbcee3d41c77de03365d233578708d23f8d66eb24c9883b7",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a76eaa7cd3fc9cefbe0e389393350179329e967551471593b5a854ad5e30c12c",
    "sha256" + debug_suffix: "c24dec435346f51439dfee8306b4d6ce7ee25dc3a94a14207264ce28e64e428d",
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
