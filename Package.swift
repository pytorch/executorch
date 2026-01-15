// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260115"
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
    "sha256": "4ebd18436f5b3bdc777b2051d7919556f09e6c7fad126aefd53400b71aacb8a8",
    "sha256" + debug_suffix: "86cd886c7ede6c0d653f35eeccb995d22a1fb18ebc237bcb70be84f180d133a3",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "fa717694e22a3189f9322b7415af56216329dde93e312548a868378a1fff8ca0",
    "sha256" + debug_suffix: "6e3693704f81808089a4445bcb2f6f55544baa967ec1eecdd6f514867ad0d875",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "27a978285b0a6a4c41052490450954af9cf66189d265d167799087a7ff494668",
    "sha256" + debug_suffix: "7e8e870d91ee8373721fb0831885195518361f87be0c25f8111045121fe60b8a",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "6b79f293da8a35926ed9a5af7c51136507f18cc410ef9384d1f5b3842393376a",
    "sha256" + debug_suffix: "aa2b23d918a69305c4348085cb86ba3f5da8dda33845fe812beb586c50d96262",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "bc184deaa79e375cf618aa42b020bab608d387eeff887a163aca9069c0b958f9",
    "sha256" + debug_suffix: "873e02d9d2caf877e7f79ec89467e2d09bbad63a06730a0f077483140afd285d",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "316f0b82c23428c99e349c72ac622484e5a98ead844a94d1dcc1c35ebe11bb03",
    "sha256" + debug_suffix: "277d0f85f9c00b29d92998a5a759877ef9356faef352630206519e2ea8a21236",
  ],
  "kernels_optimized": [
    "sha256": "5acb1372abd45c241e5f5870f56d4d22aaea93d54bdf6dba758f748822fbf8c7",
    "sha256" + debug_suffix: "329b6a9d099b8909000b0c4c5b08d0567b1dccaae2078de14f7bea44b7abd224",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "8c0d79edb075bbe850dd0489159b69156826f744767bc09906df9885b0026427",
    "sha256" + debug_suffix: "d618d73e09bae75b06eeff9da082f153dcc55bc791ea75891310326edf666710",
  ],
  "kernels_torchao": [
    "sha256": "f673a45a729d8237db2e80c9497d0e4cd8fe1a2ce68d3f3ece26059ac86eb22c",
    "sha256" + debug_suffix: "102d68f00dea6a96398520009438c60a2776cbee6b566df1a18e654cdbc095ea",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "267771daa52685a5a32590666baf4f3b4d8e88acf2dd689d3e676d3811bac86c",
    "sha256" + debug_suffix: "ac5b83e94aec0b95baf685f8b89979c7c3edb3f3621de2b20e21df02528d483b",
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
