// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260211"
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
    "sha256": "d9d514948a8e088fb473244421f2b48918f8650922c084ede97c76fb9b005d16",
    "sha256" + debug_suffix: "1d415c4d97ff5984ff12d97cf3b4d42b8c50e471847c32a89cae0a66f3770d06",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b9a299523266ec5c8b75b3986f45db4cb692a0c5a0c4c672a031e6b8694e8516",
    "sha256" + debug_suffix: "37ae1251de6e94ac4eeb858a511fd82041462b6d1b4d182e16ea2a28ccbda196",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5dd78edfb03a1e97d36088b295a881ee021bb0157cf75384d093fb828470ebf8",
    "sha256" + debug_suffix: "9b9aabfd22adc008bf1ddd089c1ff80ed06439e5f3db854db96d39e6b71ead1a",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "26f41aaf0da72451f88b73223aecb8b9b9278293bddefd19d93b10050f55711f",
    "sha256" + debug_suffix: "2ce83a03ed5fb3a6ab4171a1a73c471da022d850225dc9eafa44e8551129f00e",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "68d36f318583adff5e8d4a2dee329cb33833ed42e4251fb07ae5148cf567b91d",
    "sha256" + debug_suffix: "6689d9e829428299df93982d946beeb370325d846e4a8f2fc9046e4566638dfa",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "28a66ae7df9851a13da9597b55fde02eeec3d2ea2072b8e720cb551bc8ffdf5c",
    "sha256" + debug_suffix: "60338d256e2250819fb90ac01ede1cce80a4a6d0761e6cb9e05a873c86fa47b2",
  ],
  "kernels_optimized": [
    "sha256": "926446501e17ef481435f17592a59ecd835e4b3b87f67ccb752e488e429e4300",
    "sha256" + debug_suffix: "eb47b379f5f82afdf0549d8e75cadcdcca9b880ceddca90de0fda824132683db",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "e4e9b454cdb3ad3eae0e19b12ee5b1698623b55dc78a0de05e0727f9605486d9",
    "sha256" + debug_suffix: "e4ecedd78f2e1aa925c1b5c7ec0a2594976082a3e39794b4f66871dcb21ddbbc",
  ],
  "kernels_torchao": [
    "sha256": "cfef665e943d7b41c1181b12ceb0f59a5c29dbb64ebd1570a6f511c526833944",
    "sha256" + debug_suffix: "b1482749008965578e7c211c00d3931cf0c739ae502caeb40f60cc3662b70bed",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "0aeb18824231196a1595c896577e1697ceb531aa794bee63cf3099a5b6866811",
    "sha256" + debug_suffix: "34a53570fea44ac910c927ffc070be3f282d99a26007b902792a07e73d0cd3f3",
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
