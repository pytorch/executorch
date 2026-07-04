// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260704"
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
    "sha256": "53c66a83206837603948f78d868897556cfc150e8b4e92ceb8c418efbaf63561",
    "sha256" + debug_suffix: "b4cde60e74d744a2f42f6765d5a2ab7df19c708f3a56ad128229429685f5d87b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0cf8b03abb3bc7735d0d0a572d1e7f8efb004da84b87b20df725e98e398d544d",
    "sha256" + debug_suffix: "843123c1aede6133996b5187529c19100299bec7d3a5dd0ed4e0d497675481d6",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4aa8835e924f483abc133dc36908d358f4c06e632d64d86ac22edf9cde98e9f2",
    "sha256" + debug_suffix: "6e44034b01ac998f17dd4551d3417855971405a873eb0180a4ac4af02bc2a3e1",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "bc7ff58ccc59863810705b537934b3f5c0251dd275226dc85d475cd78ed4f81b",
    "sha256" + debug_suffix: "53da6d37219b950ba0ef533fd85a8b5f6afdd0ce5dfb3d83bae7f037e9349870",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "aa92fd79b0c26585854db966e7b096ec82e6ea3a3071fd2f1dc9e8ad32a07ac4",
    "sha256" + debug_suffix: "f521fafa7a22b49a4dc1b27a4da2041b02154f472a37708b454713e64bdaa938",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "4538f160c03f9e4dc4f75db42d0f5ef79f667f10d7034650cbff9fcc2d2cd3c6",
    "sha256" + debug_suffix: "d27517eb0101215627dbfd1755e6e4d36a4008ba2fab912978ca1972803ff175",
  ],
  "kernels_optimized": [
    "sha256": "3a29e68854dbec614a5bda9f0c88e6344cba239599507e77f2889e7a50b081f1",
    "sha256" + debug_suffix: "d44f5b949ce9efae42365ed6307f512b7d892d4c5c13a6a103c1b6938a361f15",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "79f15e0e88c4a4b7742f368da073c1c493243843ca3786c2c924dfacd0221013",
    "sha256" + debug_suffix: "8e2eabb91fa0a65374f0f9454cdd7ef52bcdd295d91b442444f5f4f68d1ca25b",
  ],
  "kernels_torchao": [
    "sha256": "bfb99abfb6fcbd07b3c8d4221b96e04fde47e5445f8fa0c3d7bcda7cb213f162",
    "sha256" + debug_suffix: "8008da71746279c193787b3e092f6d4634ae07a6266119a10c2da4849108a597",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a5a07ea7e61512a9724112baa51d8e1e6621e513eb9b256649ddf7077cd358e6",
    "sha256" + debug_suffix: "591b34b9cfeacfbcb1a640d8684f91a15abc85342846158eeeea52dc069cd8da",
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
