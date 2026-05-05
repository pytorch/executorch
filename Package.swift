// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260505"
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
    "sha256": "bd758bb4fe28fede91a70e2f67c549ddde3c65c444361e599592a0804d648f1c",
    "sha256" + debug_suffix: "e1e87cbc94ecaeb4095f5caa3621d7df81aaee5fa009e6ebd5ece33085f30cc2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3495a4dcb238a631be8fb04dacb916b7602899eaa23c29eb3479dc23cbe3e207",
    "sha256" + debug_suffix: "8b8c37c39af3da69ecfab0d08c4bd80c8fb722c18a68eb9b823d00297b57c396",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b5ae9069be20a33c7beb26f7c2c4721a455c81a8981508835b5651cf657af0b0",
    "sha256" + debug_suffix: "f1941546875ae0ee75c703bbfbb5f1aad68a282bee3849dfd4e3a5d6526b9273",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "2998edbacb5d5b48e648b649ebb4c753ee51af12a4f05ed999e56f4240054999",
    "sha256" + debug_suffix: "70d0ee7658bb50b11dae6bf934065c2e16170b4ddfa5f14629c60e059a7771dd",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "66925607291df6bd1c9f9112195332bd65966811a3a18fbf9be47bb637cf234f",
    "sha256" + debug_suffix: "1ec319ded7f316a554470f09c6531d34f6c315fc444f77fb0fa8a3614de8a4e2",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "2e733c27f508d004867cf6e9f04ffb33b6ffb5d5991569a3fcb5bc8adbf74abb",
    "sha256" + debug_suffix: "2347c0178d52fe1a588c51aeed44b2c3f1f01e9357d05feff73fc75f3f7d784a",
  ],
  "kernels_optimized": [
    "sha256": "b6ed463e91310676d2e0c8db0f5bd26ce068e04d4f9f80f7c2173d643137ec13",
    "sha256" + debug_suffix: "edde69ba771333ad7b0be9e37fa6ee62700b44ecf93e4007748fa4b63ab9259e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "e1d1f969eef663e8c46f5290b5baf99152ad0465394fdf8aba2bc8b09d68a934",
    "sha256" + debug_suffix: "5f0269110faf40da2b95222c25ba6c7dbce70abe7b44533d530288ccfcc5c418",
  ],
  "kernels_torchao": [
    "sha256": "6bf840119a29bea7697a5a0846aeb8f6ea940ff8af5ec4e7c7b581eab4a740da",
    "sha256" + debug_suffix: "9b9f0fd5607285c3f57c57f8c7b95c041b759d284958c299ae089b4a447361a6",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a7586709415e2011223a14db4b3046c3f24854fb2e5704851ba380938ac3088e",
    "sha256" + debug_suffix: "b273495e495f6919131bf37ccb129a8c3e64927f095fe417199473c0087922ee",
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
