// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260319"
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
    "sha256": "4d8f801cc392153749f93e64daf4c41b04833c0564ce284392b7ea9020539920",
    "sha256" + debug_suffix: "f11b29f69bdb7a919047bf097b55903b77fcc871cdf3ff490abdbe7f56c1b14b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e183bd44d8de8a182161186f35b80ba39a6311d22296b33727218e65b7c1c715",
    "sha256" + debug_suffix: "ae950a027637881200fbdee645d68e10e11bf3fc83984afefd37b90c1c7f35ac",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "56f9d83987907423bbadb2be7d87e0da713dd39a32ab6df98b3a3d12e65a1764",
    "sha256" + debug_suffix: "1524cf41ab5ac597d72d8abf1baa8b174dd7d18e4ca6a827dd858aca0758bfb5",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "125878c981b7852c294935851b406c1f1497c4dff57d848517a69b73789448f3",
    "sha256" + debug_suffix: "83ee1d3a99608481c9172a585dfa80bb3d3d2d908eb4869be4e1ab3068089196",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "89a57c21ad2f73ab455dfe0b45ae5cf57cff1fb49104be06d11af6ff8f8b5a93",
    "sha256" + debug_suffix: "b9c6f5f907928867e42002c12dbd0032dcfbce9df6badd6798cabb89d2bfc255",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "3a47b259c55248f807e7558e9770de56191b3cdb247c133648ec574d205005bf",
    "sha256" + debug_suffix: "54ee2457ea5712b06d5209f3c6c4ed106def07d0eb73f57640dcf91e66b8b994",
  ],
  "kernels_optimized": [
    "sha256": "eec000e4417751792fe6bb4edfd233ba7c027d80f9ae8841d39fa7e2817b061b",
    "sha256" + debug_suffix: "bd8d8f579d919edb33772570701d0f90dc0741637e226c2170ad9d175311e178",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d0ba5b2843711293d497853d8215efa179f76a848d6655dca2ae43514333008d",
    "sha256" + debug_suffix: "07f373407d4f53143839cb5409aaef939cd068fbfa3e8fe475806d163aac4299",
  ],
  "kernels_torchao": [
    "sha256": "90d53adc47c88afe61a8d4fcf831ba50b581edb58c7c8723b56ef7ecbd5c8ff2",
    "sha256" + debug_suffix: "55a32177c1643a65d0a829416960b56f113963b31e9b818a3637980faf6762f8",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "f7440d337a45034c67fb455fc47cec692d0acd27540e7aa2857b7ca36521b4bb",
    "sha256" + debug_suffix: "02b89119f90d1855dd0cdb28b712fce7589bbebf4cda3fab71d03b693184860f",
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
