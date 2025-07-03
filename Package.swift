// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250703"
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
    "sha256": "1b974d8e021c2a971ba5176afdcc7762220999c171d2a11a62581f4c8ed94a09",
    "sha256" + debug_suffix: "f020bd0ebf2db62d9feba46f67fff4f655e8177ad840af98b7295a6bfe18ea56",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "432d55583faa55692d40f55b8564f04778a90c46b83d234f9d755e053832c6fe",
    "sha256" + debug_suffix: "687cf8c5a4f775cab59ecb2c4182381f8aa622596ec95daad7dd4a96b5ee1cfd",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "66115fc31a08bd3e9b9d6675bbd6a6ad8de2a5d962298c23b6277951ead74c73",
    "sha256" + debug_suffix: "c78f57894ccbb50b9ea5fde22cc31014a1e754d25584de1ff0db5baf84ed0698",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "2759e39699edeebc04fc69af99b3b63a33b468770222cd20a7e8570b73325710",
    "sha256" + debug_suffix: "50f177ab35c0a3bc5ddd25eb7a8d0c07e809dff2c0a170b9335bf3c9cfc26eee",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "f581178e3c391e5a585390f7c96ad486de91e0555a43a34459a2a05564f9110c",
    "sha256" + debug_suffix: "e0dd221263219586fdae997353517c8ca72cc475bc53080444e15be91154c129",
  ],
  "kernels_optimized": [
    "sha256": "2b19bd57834470768582dfa591c21e87a6d7ef5fc493bb0fa49a45be211f3b39",
    "sha256" + debug_suffix: "b8e22194fca04c8dddb38a34dc2ac6a2849e072c48001ab71ac037e0362b2bd2",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "09aaf351634c4b5138af8fe53ae4294a9587277e2f3d22dd5fb822c523929e44",
    "sha256" + debug_suffix: "09150244652f0a7d5c1c5c1dd1de1f3b4ddac15d2fdb44bad112c92e6d1e10ad",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e47f6b3c5c0e60037cd6dba2872f7a77e78719e9e6bd64e5ca32d7c9edc720dc",
    "sha256" + debug_suffix: "883a35fc15aff6bfca3ecab58bd4c7fb9d007463b949313e77a4bccf2e4eb4d5",
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
