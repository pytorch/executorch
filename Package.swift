// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250718"
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
    "sha256": "a4cbf513a2b60741a16997d682dd7fa08f0ee06d1cf06843ba4253fda9d16a91",
    "sha256" + debug_suffix: "cbdb6ac58cb47997c04225d68d84708a5c76b050094ba29af61e73f5aef0b975",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "46afbf72d32b262727dca08ce43ad0fd2a54ce28811bffba8c84776d2b806f1e",
    "sha256" + debug_suffix: "b033071576fa9c6a753afcebeb2cf57c83ac2477ac32063b2d655857434b7009",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c5c89e8473b4ff8cc3cc6f225b9b031aeff2517c51b956bd488458740cfd2943",
    "sha256" + debug_suffix: "04aa674f3a713e221d382ff0cdf0aa2ddd3d5de7c50cef8a8fe658f4ae29cd9f",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0f93d45c8a8de2f5eed6bd2913a22a39f10705a4d5a5b7b02f384072bd65c3d2",
    "sha256" + debug_suffix: "6bb1598fbdf05cc664d37a6e75723bb95740224bd20b0fcdec51025966d2ece3",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "8e05cce5f10c708cb39b29ee93d93abc2c1f97a34164a14025aeeb2a50ac249e",
    "sha256" + debug_suffix: "737f8c07e267ba23f56959eb3840dd95e4f03a0cc1362d56b27353eb6012231f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c054ba89e8b87ccf27409ecbca578375360ae2f0ebf330f2c8d7e12374f8d24f",
    "sha256" + debug_suffix: "db268ff36ce1c044df14e342055230c2d15f60bcfab225f5f7bb84969f1bb8f7",
  ],
  "kernels_optimized": [
    "sha256": "aeb7e17de102e25f306d1f5ab1bfc4d6874120f80441bd127cbf9dfbfaf7652f",
    "sha256" + debug_suffix: "47a7110d0bceb26ee1c10f6b876a84cdce36d5535b0e22ea972c1f32c9834b31",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "79637dc153c8e9edd68ebd630be7369205b61aa7bcf4894c13fde941c4d8a908",
    "sha256" + debug_suffix: "c70fd542a4ac4592aba7da9f5d6e3ce4e1f056d60a71ab30ee3b3c47ddd0e447",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "8be5af6c7ae04982c5440832ee4fc1a37f5a298f03d5ab51faefa2769d2cc95b",
    "sha256" + debug_suffix: "1b47e067af13b697bb8bbe9384b1a5081731d74febcf3ac1f215c2a1b6772b86",
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
