// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260104"
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
    "sha256": "a861c7746a8849b8551c2fc6afd2f78fff3fd90319ba4ae857d8d4963b7465cd",
    "sha256" + debug_suffix: "d97b4f6f1c86ad0739880be1f0042b6ef3ab770db56758a8232254922699bf8c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1161aab24b6327a3743192e5aecdb225d1f724209664da18d7121d4f7fcfead1",
    "sha256" + debug_suffix: "b57233f58663ab62c3d4bfeda80e57949eba765a98d33e76066406bb815ee365",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "acc96216edc198259368c4c0c9b5c835fab8a63e1f67f6a86d28ea9b88215439",
    "sha256" + debug_suffix: "c3cbd04c8ec785546641a5638096a6f5d0fd5699c6daa9d309f7f4ac37ac9025",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "10991953f62b324aca4653857933465ea0b5d15f671a1e3ff48be3d9c9804e4f",
    "sha256" + debug_suffix: "5f2f62adc1a84475ce28e277eb6d3266430ab93abb6aba2199da597b3cc9f96b",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a92e8b29a74ff3592af7a5c1e97778c4c6b47c73f41d526f7e670c9c26c28195",
    "sha256" + debug_suffix: "3e4d7161b0f1cdb4c4570d56925b6386d97887a19b42026676fae36d6aa6bfe1",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "cfdca3093a91b1ea02c2185a9b0c1eb1b75c19668c5133738edb28938f52d16b",
    "sha256" + debug_suffix: "66a2e89836add3715a536d03dffa8259bcd4635f1cebd7eed072d8ae51e52bf7",
  ],
  "kernels_optimized": [
    "sha256": "ca8fa158fc8bb05d7894a92b26e15b8861c32cbf6787b14ef98bbe538039adba",
    "sha256" + debug_suffix: "e7b5fc2af1b89b485d827e5baa12ec53965f780f1393ee7249b1d4c2b37a3e1c",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "eb1b729fdcd440e473354f795a1efc24b08320fec7403678ea208582e77f670c",
    "sha256" + debug_suffix: "e0d84a09fc5d5c5dcc07529671dd8a491435d2c0631a407bba483f0e99132f79",
  ],
  "kernels_torchao": [
    "sha256": "e361b17e35503c36402290bfa834e01d454d83b826282c0d0ec410b419b00e7a",
    "sha256" + debug_suffix: "0cdd7ce6f61df47bb2d46b4e335ddce91398998b6b5f1205cd54d5cc93645c0b",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "02429614bb088b0ce9d7b2b97c382c5023f80d32daa09b5f527d893c160c5421",
    "sha256" + debug_suffix: "fe12fe86261032958cb4461741b9683230fcc875e8ad1887466a0dd9f5b7ca46",
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
