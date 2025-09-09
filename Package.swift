// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250909"
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
    "sha256": "82caa89c620dfb793ef29f4e9fa2abf286c5b87111e04837382c73688b4ada63",
    "sha256" + debug_suffix: "ee65aac2ea9c1bec2c0905601aede69b30a9ae713acd84fa9c98e1db2d8333a2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "857a740d80f5fd1b596f54e049ec2d31ea36e2a7f7dde07b2269ccfe6e5747e2",
    "sha256" + debug_suffix: "0c7cf89fbe8358dd9bd6d9f048879e985c38f2eff6f3bc39ecbee296a32ee117",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "71eabfff109242739382beb0ee55d3928cee38941c4a1e4523b8c547df55f7bd",
    "sha256" + debug_suffix: "c7b9eb20fbbb395518d3343b24975717ab751a635296ea6b9e28f03371484f07",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "79cf0227baff0bb4b2bec5dba9592bb4fc5f053be04074d087a4160102a6c2be",
    "sha256" + debug_suffix: "8862fd1785ff697b151eb0dd32a321967f2e3deec4a56bc2b83f5624d926d693",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e9fe262c29a1ffafcb38f323988807909b666b435e67ec5411bd8d8021e71e4d",
    "sha256" + debug_suffix: "82a5f8959d7a5a587d2f18ff9d68020f2cf6feb0a75f67eb4e7ff8ba957afd86",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "6890ba31fb43cb4b543b42a67743df9d0b81a9f86c9eb25bddafbb34417d8660",
    "sha256" + debug_suffix: "22615ede349bfa551f7b3f4531d421e7ee6a8043ebc811e6a82a2ddc44754da5",
  ],
  "kernels_optimized": [
    "sha256": "1c6497dc5b544ad111c0d92b5d8728504673cbfd6c2b14f47d2b5e0ab28083ef",
    "sha256" + debug_suffix: "ea99ae6d13b39f80f50e6ea05ad922aaff9588d21696a60fa77a303babedf370",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f7e4ef9df2df4b651faaf52a80efaebad883e39a5f9e798b63076e963aebdb0e",
    "sha256" + debug_suffix: "504ea944b70492b40f9ef89322eb88e1cd7754b87ea0ba4cce7bf40598fa91af",
  ],
  "kernels_torchao": [
    "sha256": "e8150a93f10e624d3461ab686684b4f3a3466a0243e55b185404b13a196b4aef",
    "sha256" + debug_suffix: "eb67dd46f57f3e1f28b2ab5febb8365de72ea733f0a7f154f1c24924fd8bb771",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "8a0afcb7544e5b2511ef546e4ee83f2874d6525984e4c5bf173bfbeb872c6a24",
    "sha256" + debug_suffix: "357757e60d8814bb81ad7670efff63cf85f60fc9bd3db1093a2d9922233948d0",
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
