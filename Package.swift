// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250803"
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
    "sha256": "e55e845f44340ea85eacb1cd8611f0e5be2385c167bcfc8a4f6f5293c266cf96",
    "sha256" + debug_suffix: "30266131e329a10ea8f3b15a83c5e3805b898f4ec689839b87a935a31756cd62",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "261ae9db849d019f5543017f7452fc063b6000934d125de29c9dee0951280d02",
    "sha256" + debug_suffix: "2fb4b42c17a317af9d0c89bea32892f3cd0962b62ff59dd4ffa90ac836793c75",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "fd08f6d9cc28bdcd0455d2017c97c553283d663b455988d14e7179d1978b3038",
    "sha256" + debug_suffix: "0205b397d27f37523660a0a9ffb919173a943ff36735ebc11091d8a46baaa0c2",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d867e5f4ec3e74d951708207930c3b9f155f57236c2b84ab4bd9fc9d70e80027",
    "sha256" + debug_suffix: "d6267e784ac19093d6c09137d1b8c33684a7a3570555f15abe461c06492d8ef3",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "63fca1c26b7e89174c9363f431da83faaad872a0ed7292fdad899b0369f5f028",
    "sha256" + debug_suffix: "3c7404c3de31bf8911291284621d710a9afaf6f35efa73034a1b4b172da97f86",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "064638c282e60dc743251718ed6a31044f8f221b76627bb7af69307618af72b7",
    "sha256" + debug_suffix: "dcbe50105cabb54ada39ef9b079150164a608e8875d157edbe94fb6e9d65e393",
  ],
  "kernels_optimized": [
    "sha256": "022f3cd022c0802d3b25c8d6d203f7a0b07cb1184d33656db49b283db91cdadf",
    "sha256" + debug_suffix: "e2e16806181bebd64942cbd4905536ad068f077ee52c88fcc071fa45cfc66760",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "3ff9a62de2017edba88b3f9ef821d38eae909345aa12fc9541b9d140046da886",
    "sha256" + debug_suffix: "be4c3c0634e4db3086d2ea0d0f72fdd25473dd31ef1ea1a2e5bb5adff5237dc2",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "42621078201a1118cdf1950f0a0c19ceb5eb5a3bd97bd9836e8f104fc93e4578",
    "sha256" + debug_suffix: "383cd9f73853e4c2cb2950b24533188a6293e2845a54e0a047dd52a620129ac9",
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
