// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250622"
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
    "sha256": "484293a345d009c5adf897fd247331e08e8db8018ca3f013e80e6dd49fabdc83",
    "sha256" + debug_suffix: "97abafa0d6a374e2cee8df5251c1753b7b447ba2950fae94f1108f66ff78f7cc",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "478533af14ce12e74605e6cf906bcc5fbfc0d8a8a283b7e0ea5e3879c89cdc36",
    "sha256" + debug_suffix: "e0e5a016a8cf1c30ab46f7d25c9bcdff2012e8e0bdd3e5013c8e3d29324d8d2b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "63dbf2c32c14a0e8e82f9487b8a38408a7d50c1fd02474b20fb811973366e470",
    "sha256" + debug_suffix: "c0b841f8c5e05a06f59ada1111fe11c8bb361e66a88c616d056fef2e828191ee",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "3463e542b8aaf4f2970c7b9f878e2533bfafc0fa52d17087f0465b359ec500fc",
    "sha256" + debug_suffix: "d40583bfbbc6e00f10e856adc2c2f11d1300950771a4a03397310711a91e2026",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "967c13e3ce3fe01e7383d5b324abbc9e88953771052e2dfdcf6f26c7326e0167",
    "sha256" + debug_suffix: "c6cc8b0404caf2cbe910f54d9d6ce641eb05aafa815bcff56fc9a44db3a2ebaa",
  ],
  "kernels_optimized": [
    "sha256": "f9c5f80aea768443c476790aba0ae53d5221c882f2c1d4b01eae2f2ed7374265",
    "sha256" + debug_suffix: "e326051e5cb5ab3ccf9b56a7438c765a2ee7fae70f30e8d981b5c163aa6b3aaf",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9e45c774312249c6ac72d17f024837c0a8bdd6a4e1cde7d0858a0246afab5767",
    "sha256" + debug_suffix: "79bb15fa377865ce905485a15bac5cda590f5cee9d988e32e3b3e55ccba4d4fe",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "4391930171715916cecfd1e00850c5827e6e4ef9caad7d3d6cd38c5c8369c3b3",
    "sha256" + debug_suffix: "2b9e1838b7f528cc93fb2591774786ab7bbb49e8340c46a01daf7578b1368740",
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
