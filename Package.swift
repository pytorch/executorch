// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0"
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
    "sha256": "934f4c320f78d715425e17a02afb30b89f4e5bc594d9de96e7141580a397f8cc",
    "sha256" + debug_suffix: "4bc1d93bbc1c33bff8700374bea8e8419af10a3bafb740616315df8cbe957170",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "00cf16e85ff582af5b9b0a8f05049d95dd5c7bd0d416ca22246b3e103d0dc0c9",
    "sha256" + debug_suffix: "b8ba909d8cd4ea8e28220e5f9ae6b2ca70271cf87b95f129e26c51eeeb9cffee",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6123bb64b1aad6f6dd8ffbfd4c0a396ecc053f320d4c116f98ce976a4e9a3952",
    "sha256" + debug_suffix: "deff6b2b1b061e770719cd3180abd49473c7d7120e446452beebb53cc0066adf",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "3eeb1e4b5a1b5fdad726e98b00bdf3fa132be1f5f6eb652003c798d8ab5c5109",
    "sha256" + debug_suffix: "931da2a0abc7028084ab0a7df4ab6c384327aa343dc75654b387caae619a4d78",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "16c7220c0d1ad8d7eb6645ce680bb6eaea2a448abc82095632dbb1ebdb0ceff8",
    "sha256" + debug_suffix: "78f4412de3e7ebfc66ce2bcde2a6ef5cde09002d08a9a37dac8b1a5c241ee20e",
  ],
  "kernels_optimized": [
    "sha256": "2fdb79337926c05bde421a22297f8bd91fa5fd63507301c0e55d50998564e7dd",
    "sha256" + debug_suffix: "37a7ab123a7bd5b7f9f080495ff72ef8b5bbcac964788cc4b8d312092e395eb7",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "4fe18c767fed3a672256903ad43871e0f319f4060d42fa5ed4f35554f7865619",
    "sha256" + debug_suffix: "734886e115b95aa0903e5343e2257828f7ea296aa77de3a30f2d4273d662a51c",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "8735860fd260f8faa651f7e051c0ad8d06d23a645dd7370be8da56388123f8e5",
    "sha256" + debug_suffix: "354771388b9189fbbd34aeb2560788f5b3c8b7a6f6866ad361e00ee0f769ef31",
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
