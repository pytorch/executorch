// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251118"
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
    "sha256": "f2036c0e186b9831efb9e8706da1e4f4cd5de9d717fb8945fe93fe5c3fe19e42",
    "sha256" + debug_suffix: "30f14590ef51ed969c9c054097e3af2cc928ff3a959bbb56a2251b45dfd4dd59",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "26356f3292c98d36f4fcaffecdb0cb5b87da488a8a431a126ad06245f2c6b152",
    "sha256" + debug_suffix: "4566648c8984bb85c5bc3128e4f801887e47a7aca8f5bb768e36cb21573b7843",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "45d5db5ef4be641bc8214a5b062a51e0b2d627a577184ac4d0395a99748c8618",
    "sha256" + debug_suffix: "2ff2104779de696ac405b2a213e4ced4a607a81c8954e7ea6a788eaa0f4cdfb1",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0b17b1abdf1c43dab57811d4d271fb0dba0934c726332fbb3278fdde92d38714",
    "sha256" + debug_suffix: "b1e13597f2187efb1b396a39a8e9222af6412585d416f29d0e86ac0343c1df3f",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "62221fd4c507d71730638a6a28b17a818445741c4982df807db908f36b61a375",
    "sha256" + debug_suffix: "1f8ba15ca37fdd6181ab3e50576dc4fa193903bd3ac8178ee099d345507243fb",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "7a258bc2a175513c4490336b26d694cbc4964250643e58618052163f7154d884",
    "sha256" + debug_suffix: "887d59b04b4c34fc5a6cc1918418369c2320184f40ce6e91894c4178c64d3244",
  ],
  "kernels_optimized": [
    "sha256": "fea222e3f752688c4ee152060e9f3cf657a519e888349baab7d9dadd724c8443",
    "sha256" + debug_suffix: "939cb3a55e9d97b4da7e4cc1a297f96dbd3ba256e254412630df34a11fd04ccc",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9d2b9e214b50f67447becf1ab4185302d4baecfda22b0673045e67197c9d47c3",
    "sha256" + debug_suffix: "fc2f8eee1a5026b25efdb45430940c35abb439a4dd0b467f2f13f556bc3ab908",
  ],
  "kernels_torchao": [
    "sha256": "4737aaf4dbe4dcaf14f8abc38ec5febbc8445f0aa7b4fe64104e7c9600bd9352",
    "sha256" + debug_suffix: "4ec6e0c71c60be8a6eb768772a983e55d145778f59f299c7022e27737491b2a5",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "33daaae673db4f6f20115a86e92ee09e5f6cd50782ee0b33d26baddba196843a",
    "sha256" + debug_suffix: "67891703e791279f5186c7e3c3bd302e77758f38e5de8c56fd1873ea12f0daeb",
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
