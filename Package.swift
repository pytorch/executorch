// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250603"
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
    "sha256": "56fc48988d450d43dc98b82acbb7b73a6eafeb06ed3c6f6bcbad2f2b57330632",
    "sha256" + debug_suffix: "0ad9f18976f31d96b58c56ebf7c66d19ddd3886f41fa914761d87a3cbed26a0e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ad52f1d62f55f56f15ebd173eb63a78c55789ca5694ab0e050d4480696a883e6",
    "sha256" + debug_suffix: "09d984a8db9e4e19529f9894815ad893405cf1e710e770a94c1c10868ff143a8",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f747a7c7c709ebb44984ee143e36c44ee121cf1741c3260b86bde270bd113172",
    "sha256" + debug_suffix: "1aebe35f70a820632f79123d7d4755dbea37deee341bfb1d125888e78f2f58e3",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e5028bd0fe4a04aa03104107c48bf9a7a4e1a4e5d8455df00829bc11ebb45607",
    "sha256" + debug_suffix: "08322d7eb0e3f82bf45ab91dfb13ffcde205ade3cdd26eca1d4a51479310475e",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "18d5080b55e1e0f1f5af9d26936eb153d506064b29ecc9a3a3b88928bb50ef8e",
    "sha256" + debug_suffix: "b91f3d8b30ce15fe26c33395b1e1f1c7c7d54c2af6cd26abc0b509cc3aea5f5c",
  ],
  "kernels_optimized": [
    "sha256": "1421a6720ee02db7a0239885159b9c49a955917b223da06f0c16b0f122adc035",
    "sha256" + debug_suffix: "7254c76e74b95de609bf2334a9d9e0771e63897414f39cf73ecae0df8462b077",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "7c4f052a11aeed67fc935c11ba57a77d84c28c7ff33f0c65a1484ac1fa8dec44",
    "sha256" + debug_suffix: "fcb105b4bf6240d1e03eb503697372ee3b3508dd71f3b958c6e46c5339c01664",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "c5cf196e8b4d69384765524721ca0626083f56b828db1f60b77754b7ca3bd625",
    "sha256" + debug_suffix: "b6d3d09355722162f025ab3d00498b0ae938b243872be791a6f214706e9034bc",
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
