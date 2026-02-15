// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260215"
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
    "sha256": "3f296ecf55026a9e4498a53bf908134f4348698919a7fafe7bbb5efd0094a819",
    "sha256" + debug_suffix: "e73df36d5d2a96b724231ceb7b10834f54b96074242b48fe0bed39facbc7719a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "aaab1d767b2fba7d2007bcad446e5d212c1b3a4a9cca2146b13d2178090e22df",
    "sha256" + debug_suffix: "c00d50e346c74fa56afd061de67f2da1ca85318e7d01208899036fc9b88fdb14",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4589f4bf9101f3e3ecd2a825cc0f58af0051005bdcd3d14581fa0d0e566651f9",
    "sha256" + debug_suffix: "acde5c970089c33ae982f8a772c668473410ef37fd3af87552754e17bfa0a19f",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "8499df70dfda231e2e206405cc8421aa42b755ce8c682147e55871f012e5b741",
    "sha256" + debug_suffix: "27c25ed6723c4c7ec19f157a930c0f2c96bcfd1bf52066e23709c4106cb7be74",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a44fa7c7f861beb20c1b99fb715229119cab238d41dad96b80e8853c4981959a",
    "sha256" + debug_suffix: "6eb053c77ca4447af1279b87fe7d3e51bb4ffe275ae0674cb6e1edfcf078498a",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "2514e9f27dc329a8bb83a65bf1ca9d5ceb70ce35ced809f8c301c5c146427649",
    "sha256" + debug_suffix: "4e397d8999500ad39b64c90849b1863fcbe2ac6e06401dba20b021766a61ae00",
  ],
  "kernels_optimized": [
    "sha256": "67863e5b435bc4a37d30c949f1523bd194ca88fbf7445e847ec10bd1726562d1",
    "sha256" + debug_suffix: "0685efe026b9d08c4767fc40f5b0be24f598c1a264051950fa1cd2770c0f5434",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a7fd7e594cdac83a09d98311640b5e8daabb99a72912b235f3729bd9dfa135f6",
    "sha256" + debug_suffix: "aaf7dc78d591019006bcadbcdbb293ccfc1dd674572615044f98a4db27c1629a",
  ],
  "kernels_torchao": [
    "sha256": "612eb36998da3b3551741bd5c750b770c15923f09d520aeae8d7639c2f2ccc2b",
    "sha256" + debug_suffix: "ca4298a1cf656901307c49d6bfe0a9efdbf5c499b0188f3a138225cecc6ff65e",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "1532df88b9797bb47a4f13a137f8ef4cab55f67d3fbb221887bec284aed99f27",
    "sha256" + debug_suffix: "8905d2d506c591d42dcb5af1603d4410c148a1fe08f7eccc7c69f70bbc13797f",
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
