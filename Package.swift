// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260106"
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
    "sha256": "8821dac1ee27dbd8356229a8f97233da2d5accc1c9f048cdf93f59a8d2f59190",
    "sha256" + debug_suffix: "3a8130c3513ff88725ed29f0b94627f78be15f2bcce4a517c8c289a48a4ad309",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "01ca761f884b232db367d70acecb8662c4045570d19f163bec816a793a5e799a",
    "sha256" + debug_suffix: "c3d8850b057bfbe52ed756c4cb8d926eceb9b549d02b7a9673aeb9c582c759ee",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d6d0cf404dd2f79d8dec1f35643404de092a59ea967604a3aa912124f3251b4d",
    "sha256" + debug_suffix: "24bfc892255af4837168ef642a224dc4c8287c30ae8743e4838373bfe27f9525",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "81113ca2dc5ea517ba14dead65436123eff131eac7d2eccf9ef7e3bd70d25b19",
    "sha256" + debug_suffix: "0b5a4cb4407cbeaeeb6303d9d497d4f7926e70ae88ca13b2afbe45cb4dbe5860",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e0c398da0cb93b624aa5fc00c7648975476105e27229fc30a4f524c13ef8ea6d",
    "sha256" + debug_suffix: "330be05693c988cee6f08d42838f7fc12f23214fdbcb0bee65ae762698e5a005",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "6cea8d322e0db00fef0741729f66e4050bf7e68ae5c9310a1eceb5b8b8f569d5",
    "sha256" + debug_suffix: "6d87f418292c1dc024c2784b380fef52f3ba850a411bbab0935e10cb3121b2ca",
  ],
  "kernels_optimized": [
    "sha256": "ce6495cc8a687a3588df02d699b0d9591f2c7b284a7aa0136daea115eb1260c9",
    "sha256" + debug_suffix: "0ae3687eb1271c0cf668701739696669d3447c0985a73048c30d4dca27cca10d",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ba89c3a356d05228519a6d8c3c0ab729f3ea40fa53558efbe310bbdfc4904dd5",
    "sha256" + debug_suffix: "4d2b25702efb72add789ed872caeed7b49a9e528b3431bdec2c1186d8b315e87",
  ],
  "kernels_torchao": [
    "sha256": "eca91051ac8436c0246be44b070b7549a688af718915862dd12ed42f9ff1ef53",
    "sha256" + debug_suffix: "a5df1d5efbbac2c8ad8393e134659ef41e1f73f7d0fc0f241a605e93b5c7669e",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "6945b7322bb5c12f429d215aae5cc7c05559793eb7378ff61e1141b3cdd0e76f",
    "sha256" + debug_suffix: "29e3a87c6f26ef1eb3e574aade246ccc950b088e23b9a6d312b0d4f23d8bcf4b",
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
