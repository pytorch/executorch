// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250721"
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
    "sha256": "4833390389638476e484b46c84721145fd166197e81f450660bae56ddb666d5e",
    "sha256" + debug_suffix: "0ec3bd32a88c6c5de98beaf5b0347cb7367dc960e59748211ba346785db468f9",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "85150f04ecafd812b2828de0134fd8d39ce45a29e80e81396fa6f21772ff7aa1",
    "sha256" + debug_suffix: "4aa53f476098869b2ddec3ccbb8d830c5707c35ff980ceca57a84b49f86c9342",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0608d2df69f5d279abda74bd5e0379d347957b01cd50ad6158ee79baefe99b0b",
    "sha256" + debug_suffix: "525a7a8b689393e25a61e7f7e422a08edea97905505fb4ef9a67291533ae648a",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "74627930f70f64821f6009a53a730c71070692305be902f6da570b32cdc11c2a",
    "sha256" + debug_suffix: "f8dba233db110d12a1f2286c040f01e87f6fcb7d5d2fe2662d8a2ea57a7777c5",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "ee142f228e9034376edf29d8e7b6d0bf9387af31a487749defeaf71476df4aef",
    "sha256" + debug_suffix: "7e8e7b57137bbe48f4a41826c8563be8e08067191ff068990ccae86179bbc865",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8e694f64df805cba6b7166ea4d76704028f31f29b6f4234119ecf813a6fea74b",
    "sha256" + debug_suffix: "bdb0362f6577815e81c75af7ed675d2d70b8f1b0a2f2898fdf3fad8e4f652664",
  ],
  "kernels_optimized": [
    "sha256": "5566cad7610216a930a072bc638b2a4280472ef749dbeaf1f6f6579965da4393",
    "sha256" + debug_suffix: "6bc63267e05706ad481c8318efd9b942934ab2197be1f3e41449be017b07aa36",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "87bce806b954a028e5849d1734f77a0c1c75b42589311c988d5d5ce1f3ace511",
    "sha256" + debug_suffix: "08ba4d08358594c71701681663d1258800376c1fcc5fa47dd4c2104d6f2cb528",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e96115ab302af606414bfa92a76fd70592898a088cf06f29602dc2ab99d91463",
    "sha256" + debug_suffix: "930b8bd86436528a7a385285c5839a5206f592b93389eb7de2aad14d34fa80d4",
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
