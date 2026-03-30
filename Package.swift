// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260330"
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
    "sha256": "e2bb8b21fcdf8fa7d69392b8f47abbdb55de8d4d70106f61db498e2dd8798757",
    "sha256" + debug_suffix: "fcf97b8eddf9313d1cd0717b1ee333b39424b538752a8eccf93e58274ff8732e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "909fc01d77c21758b88fe6aa07d4a6bd0c28182a2f756cca38f84e53149c9ee8",
    "sha256" + debug_suffix: "449c98c665c393381e0fd7b2183249882a3b2ec3ccb76edcfa78806ef6277801",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4ef04b36ca8421e2c20ad700f7011345096c6288981a950c1db8affccd7b6219",
    "sha256" + debug_suffix: "ae25db774393682149bfccde9652968507eeb1ff396529d845461ee077ec497d",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "4ff17d78ddd33da9a3e94e3d62a75b8f1b8c60d4020baac4f2f4c5240e993564",
    "sha256" + debug_suffix: "611f26114402b08f5e640eff0c84ec5f7330795156e92f33e49f6df45e7a9f34",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "37933f33d569d3174a6a19060e0c191e17565c86b6c8a9dbd0d0a2a6e8f1f8ff",
    "sha256" + debug_suffix: "d227e4e46e82e2a1904c42795e9c14385d2b1f12393b7aa6b092da822622550c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "4902d8392ff81ef9846178b9545ae830c13d20c90a07d893e246d98fb81d0d6d",
    "sha256" + debug_suffix: "3fafd83d0d76d3ac75589c9d91652823f60acc5ebb9d3f613055073b085f24c9",
  ],
  "kernels_optimized": [
    "sha256": "f4b92187e8df047fc1d7165cf8fcd65351df220b29057c9b7c717205004bc182",
    "sha256" + debug_suffix: "466e1bebb2ff2caea7f273180133f0cb2656e110fe749741147d66f52d875339",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ca30fb23d3746bdad7a17552780a3cd88a13afd691588f5a72a60dd50675f6ce",
    "sha256" + debug_suffix: "1c79742c1ffbfaa8b86e1602e72127bc8acafdbf2b3135975e24478b8d279d65",
  ],
  "kernels_torchao": [
    "sha256": "a298fabcbc3f35fa7fd7d0afcbeb653300ec86e16a38810182c68aed00dabf53",
    "sha256" + debug_suffix: "1df746480aec7401712542886ee51d08341a80ead70e9fc6e18a744258b4c90a",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a4da1c75ac3a0b12ddff6723e9343538630ac410052c64fe336ddd7cb20f48d8",
    "sha256" + debug_suffix: "a0519b5cf92b728d0cbaa5d1c9aa6f9db1e45fe5bba83202ef52edf405e3e690",
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
