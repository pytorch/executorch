// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260731"
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
    "sha256": "ef3901d60fe065c9629f88ac196796c3a473c8173b23e4ec0cd485ff7c4ea562",
    "sha256" + debug_suffix: "c979a48f3a91249bd209e4e674030580c4fe1e0166624c277c39be13254eccec",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "660bf12ede8240147c57221d84625ad5d8d3c516345a9b9e2fafb119da059007",
    "sha256" + debug_suffix: "f25ecc226dced7ec3c2a4c3fb69e10318fc5588ae94ddc55e22958202c6dcf7b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a46556574b2f715bf5e77475023ddddf1fe32ae00c25b2304c6a7f17a0bee75c",
    "sha256" + debug_suffix: "1e3b0fcf79d39f0e41153ef63af71701962795ba343cd6019f3132a352b1b254",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d1d0ae8c7918655e0b0781b11f839186db8b6bce248ee8aab3c2ece62fe6b494",
    "sha256" + debug_suffix: "61c21f72abe4b0cdacec2482396b827f2c2a0921595be91578473bf2b9ce7fda",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "66a1d45d3fd71a631c4770794f522eed514a6107aa065497ad0b2cfaa324797b",
    "sha256" + debug_suffix: "85f914f09065a0505c954b7acdd8a2a698ad1b3da7b610c448a5cc9dbc643fe5",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "6b7e820a816a85a78975be786fff49fecea24fb4f4e3ffe2d1e52ffa8364c4a4",
    "sha256" + debug_suffix: "aa109f722ca43fde2df47a2926fe175caf77d0e5a0cf387c22216bd2cf1461b0",
  ],
  "kernels_optimized": [
    "sha256": "ce9b904f9cb069ebca430033c475c78009923191a9f2ee73e7f0657f2f4f37e4",
    "sha256" + debug_suffix: "90473703da4e38dc8680036731d663c141609b8fc40045ef01d015afba33d34e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "3b35d5b4c95a3c0e0e3cd79a7786af4a14b183ab88bf6ac197d5b81da0206e56",
    "sha256" + debug_suffix: "e8205eb28ed9add91fcd93365446a96135a46a48d50204b7d2cebda5e8cc74da",
  ],
  "kernels_torchao": [
    "sha256": "ec253761463d2325f572e7f83381392481cd02fdc78ac2ddf236b6bcbcb859fd",
    "sha256" + debug_suffix: "bceed9aecf256d7ee5f11acd798b9de6f94a31ae80686b28af89670ad4bd53c8",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "c9a53643dcfea95e4d47978020645f3fddc97b95d06ae0a6b263dc1bdb9f8c5a",
    "sha256" + debug_suffix: "3e3ee1424caef37627a7e3eea5bc7e05bcf5daaa20bd07a23017ff24f6c97623",
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
