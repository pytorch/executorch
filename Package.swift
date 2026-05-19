// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260519"
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
    "sha256": "d5b44fad06e5354612bb37e4ed20c5599a6c182a1bc935381138d2e041ff6b27",
    "sha256" + debug_suffix: "12991dc73944390651e6c56b759ec4a55a39d4c63f1e44b0750f4b094a5cd752",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "926455ebddb807e08ce6e83cc0788cb6c16079d02e0233aef47f75d9b79aa101",
    "sha256" + debug_suffix: "4617c6e15866ae5b0d9036c3ba9d518fcfafefb1b044f6fee993c7a4f288c63c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6b94b85ff7b83e4435d2a3898e169c87498906743047fd61e80492e8ce6b83e2",
    "sha256" + debug_suffix: "109049262fe3bcea24f15cbe41af2c6f7d659ff93a3da0855a22731b2d5fd2e0",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "1f3be249032c7b806dc105945cf6d46b0c57a108b0ecd9c9c5c0bc8bfd03c7b8",
    "sha256" + debug_suffix: "bfa86fc4b2126b1fc46119f87ebe7bfa203d1069a1b52f85559b45ac1910ead1",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "9f8a6844f14f3160da02d1c4afd713bf0560d9f2c469939f50ba23aee2ee6efb",
    "sha256" + debug_suffix: "6134682d7c13926e2f5087c182871840fe16679f6009ed34802383caca16fe62",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "ff78f3096bd7c601a8f8368f63c3479f1f2063cf9eb9bbcdbe2466b6ddba8400",
    "sha256" + debug_suffix: "2bb9d91d3c2ff941350f97d03f75ea755f952cf284b563e2dccd87af14c07168",
  ],
  "kernels_optimized": [
    "sha256": "26b77eef5434c6f9cd2dd6f3e4372c128a248bbe6030f39e7493be9724ac5860",
    "sha256" + debug_suffix: "d532f3d2e80dc776bee3986658a92a28ff41cbda2e92cceb1a5692d3dd39da79",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "3ef640352ca470833fe9eb6cb294b4a4b5ba1272bab7c076ed4e321ba890ad62",
    "sha256" + debug_suffix: "2e497b7bbc1a9b540860c797f45fe81d21d9187015e10785f5606f90ac9f0885",
  ],
  "kernels_torchao": [
    "sha256": "f1b35227b757912197a17f0d4c790b2e5953194c019c18f3a8cd696a34b1bd9f",
    "sha256" + debug_suffix: "153313a761163b7a4c8c4466329326ef3e7489cadc2def02b1e6778a5b406210",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "0e1e8b0aef9c9f7a52cdf360d759809db062a9b2b0212c2fcebb3d5af9eee1c5",
    "sha256" + debug_suffix: "1465834e90c2c504517ae4ef90552e5ef1397204e85cd9e3cfc3d9ec12dd8c7e",
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
