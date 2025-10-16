// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.0.0"
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
    "sha256": "5b89c4e5cf0afa791a55a15764155dbff1956b84a4ff92e673e45110dea557e4",
    "sha256" + debug_suffix: "cf1c70b9c7d86337e66ce79702a7dd464bd9df708bd9c1f604b413a6c9cab478",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "f55bc639f74d29bf6cff8b9761297251e470b1802a1e3f63ffe90dc76f1b8857",
    "sha256" + debug_suffix: "9794fc935141d984badaacd2ab3ab91bbcf52161e4aeca29225358ee2d103084",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a9bc35784a84fd0ce5f579235e74b997684f15a2f93c53cf2e3fbfeefe24234c",
    "sha256" + debug_suffix: "ccb16482c616f6763ba12b79d7b416251f0d384a17dfd3aa6707c369194ead8f",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "95c2f2b60348b86829e9ab76b038cef1dc14a857a4771cc593bb962685cfda86",
    "sha256" + debug_suffix: "74bb8d0adeaaebe83da9858ad2844a7325c590cf36919b3f12f6ab41b7e10796",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "c9d52aac7cc329895fe16ba7206bee16e7213a8827903457b131f79ad5097c3a",
    "sha256" + debug_suffix: "ebe8ca8d1f7f99dd3c308d4f5586f62bad9a5de7f71f26139249f6f5948fd9e2",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "0068c2bc6a1db55ff7d6e1728370f345deb60c7b94c201c218232ead5a1baf21",
    "sha256" + debug_suffix: "43b2de8be5d9a52a230f8064d1477750fdbe2c981fd275b45bb54e372ecda75a",
  ],
  "kernels_optimized": [
    "sha256": "59317512964fb755f82daeb31ee7c04bd55de905640a3f5c9d0833aa520b0baa",
    "sha256" + debug_suffix: "7a08c5eba57543127402ce2570d5d3458015ed8b557a86a127b7b0fcea3a7d61",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d893406bc14db1df917e2e14367e0554ae055c79569a1b31604cf5e38c8e8fe2",
    "sha256" + debug_suffix: "fefb462116d36d0b0f5b847c1ab5aa7db05519262d24dc4adc57bcd827140a89",
  ],
  "kernels_torchao": [
    "sha256": "ba05cf451b1ffdd3778589c0f16072e29bfb692651b36e4ee76f32d24d6b81d9",
    "sha256" + debug_suffix: "621cc7b11463dd82e1fe109e9a73ccde97886f8e768a6a7ee108a80104287786",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "c35486e61b9c7572153ad8b26053086fcbeb7c5a5a869afd70d8eaaa9c1d27f2",
    "sha256" + debug_suffix: "e1f30042dd1983aae271c67fec25f62bb196e9216bd1afdf32a629bd71d57abf",
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
