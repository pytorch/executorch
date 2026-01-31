// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260131"
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
    "sha256": "a3eba740014eb128928e6fb6ab23313cae33792e6ad434394b5dd12f75769e39",
    "sha256" + debug_suffix: "984c6eab8205ef5ec9f49d10fda09015791bbb70285af11cabaaaad08429bb44",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "f2ab0413880cb9b4d7dbac251464198ba3c3e907a905a1a752e7f4703ca170c6",
    "sha256" + debug_suffix: "e9859f0db379c8cc52f2826f04f66dec9f802487e40910ea785a855ece7c1b41",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "040e8edccacfce34a3ff01981edc83cc6910070b01e3ee98bab9ef426b7745e7",
    "sha256" + debug_suffix: "9c1908c77febf528931350f78dba0ed9dbe83af462f89d93a7bd0fd45406d32c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9bfe75a7157600351fd503116b5a2533f073bf3f92c7d4d33a67819cc59b6cf7",
    "sha256" + debug_suffix: "905ebaabd2297791f3782c7398354de32afcd084741dc0dab4c64410b055979b",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "4130eba644516fcfb258f6e24dd6abd6e9ba4b1f5999bd888f9fb8e259fc2a65",
    "sha256" + debug_suffix: "7f579427880c6412483a896ac25ee5968550e7e488644a47396f6db98eec7b0d",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "0605a7e09f70853f9475ca8c9a56d42d85cfecae7188ec3179856cac21ee1f97",
    "sha256" + debug_suffix: "21499d660ea119b5a69ebdf41bbd6e14781a47db8bc11c63d76d443f824bb080",
  ],
  "kernels_optimized": [
    "sha256": "6d27e248f1bfdc8de8646d5797ab615e5c100c715a8cc1d7068f129a17e0b243",
    "sha256" + debug_suffix: "23fc296d632e6fc28cbd25771b1149778e5d39ec7c6a9ea0c9b9f081d59bb624",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9ffc75b0f3ea1515535e9da0bfb19a5e02625abd2e042959942191262aa3c979",
    "sha256" + debug_suffix: "3dc18f2651a2f0cf6bd0d54263eb4c9ad3249a4c091349a93353957ef4bb092f",
  ],
  "kernels_torchao": [
    "sha256": "4790acb45b6777cbb7d933464fe188d6d27b8a596596cc3bfd57b5aba4bb3012",
    "sha256" + debug_suffix: "597f33e0baeae2d6e33ba08a4a9b7e3c95b233728fca1c70537a7f6b9df08cfc",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "64fd384b666e7941b0e9ecd42b308d062d120a5ac9b28eab7b3be229579c1119",
    "sha256" + debug_suffix: "58a42054ce6bda13b7bbc2edba2e82a45ccb11be8365e868a40f995dda149538",
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
