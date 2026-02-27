// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260227"
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
    "sha256": "972912d5e9ef6932222d96ee8c869a08aa504f2217b62730fb08edeca770b633",
    "sha256" + debug_suffix: "503d88398ce1ccfd8e650d70f703b842831565e656acf6aa12f5cae91663cdd6",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b704a2fcaa2c9687af3a80a26662c622e998f4924f8140c05de11db28ef12f00",
    "sha256" + debug_suffix: "1bc43b4694098d9376901097c9eab052059fc11914b4299766e729557e2190a5",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6ebe21a42fefe57bddd9388b7de726bf2a546e337f5a047368b9fdd670c50d8a",
    "sha256" + debug_suffix: "70734453b2fb4dbcf9985efec87e73afb86ec79a458d83e879391d98fee73426",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "1777f14b1dc762d0b3157dad5bf1b29b89804672c86d3af184aaf36f330d179e",
    "sha256" + debug_suffix: "130a5ed0511135a583472fbceca287d8c14dc3a1f3870a6f2b69308506db73c6",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "2558f9844467837094b2a80fb859dfe70ccec45f42c3c74d997ba31fec1c1c18",
    "sha256" + debug_suffix: "942a4e994d0c20b519baf121b024b9fd179eaafd2a4f76ca6a69cca96d36fd48",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f339dca5033e90ad44446d7ad59b4916e73f3a5d3a4e4b32e34299a9320d20a3",
    "sha256" + debug_suffix: "50e839565f26da4b60ac68235b354ca49263e47f6e6ee734cb274d6da22900df",
  ],
  "kernels_optimized": [
    "sha256": "038ae51a045bac88015b730d896b892366a853f4fc35cd75e749a1f282a9ccf2",
    "sha256" + debug_suffix: "e7239136351e7b186bb24630c423e45f4cdaabbb91c31aba2dcb30ad3a5be914",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "bb8009a9fc1a69e4fd5f443d431acd0f310d146bd04a7f78132a72b967067e69",
    "sha256" + debug_suffix: "8b85d8387994a49066ec1ef9413b09f03a41207563a0730460c8e9357ff9703f",
  ],
  "kernels_torchao": [
    "sha256": "76cace35fb570679a3f5cccd096559ca71e11929f6e9960df3cb00fe1f310382",
    "sha256" + debug_suffix: "27c1bb3c2648b1deee2b6b080d3dd46f7fe46ff9f632eebb849afd0b8eec1acc",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a1704be0f21e221aafa1109938331d81504e329f03d46e72bee46c6387338bc1",
    "sha256" + debug_suffix: "1266153e64ffcbdf6b41f4e8e44398b778d45c91d5af518cf20e5302e69c53a9",
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
