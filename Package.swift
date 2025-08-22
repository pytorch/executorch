// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250822"
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
    "sha256": "5bf332de0c4dda88c5b1a33704d2dc4b6b74d9f4b213b6cda15c55f8ece33caf",
    "sha256" + debug_suffix: "ba9de0d8b3e4b96438a84e77e479ca0831be2d4889e1ec1e5229e49a362631b2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "376f5c57b0bf3bd053b1c9bbcf9189fd2b45b13c949e5250aaab07492a0b993c",
    "sha256" + debug_suffix: "aba5d860a994d75246258af1ed3069dddcf1dab7accf0012aa97a1826c9a26bd",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c150ea13498785336817f936ebdbd0951b04418e64efb5d6ca7bde137ef535b4",
    "sha256" + debug_suffix: "f566508d6f9a0325542db26cdb6e6bdf59d2b8af82b2016e6baac25e916c6cac",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "50de493430a4b264e1b364f67e93ebb8b4ef1d923fb6a1df9a2bc85cecdbff9f",
    "sha256" + debug_suffix: "8ec830974b121de96323aa007bea3270dd2c3a5470b654b759a41320822bc11f",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "0487137d3e75d9919a7503613db171a5d689113500d14c9083af194159df5f3e",
    "sha256" + debug_suffix: "e83353c640093d2a2d783e032f06c7342a4ded22f93c6ae9b090103042d072d7",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "972e77b1c4f1b40c7b82291587762b10dc43164a709418da1254f61d9e41e9b5",
    "sha256" + debug_suffix: "f31643816353ebfb46674811821ca757f1e7547a35982f45debf725d18844df5",
  ],
  "kernels_optimized": [
    "sha256": "511b3e2aba6cf2f9ba03e691a825d6c712768f6e82f36aa2bd399eccd9e7efcc",
    "sha256" + debug_suffix: "128069a65706458dab7ddc113ed778bf74434539db362e80c208b0fd106b8cb4",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9ce4b6603f99a4a29618894efae25a9d100ced8bed8fcfe6736fee13ec19e661",
    "sha256" + debug_suffix: "016b9e026144a27aebdcd839a2c223501ce4b659bd60ce139b52f018f994fcf4",
  ],
  "kernels_torchao": [
    "sha256": "__SHA256_kernels_torchao__",
    "sha256" + debug_suffix: "__SHA256_kernels_torchao_debug__",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "bb1afa470291a7d7f31b9d6a61d18bc98692304dc07f0ce30a2e117191f67430",
    "sha256" + debug_suffix: "5448fc85e8dac8eff52a7920c9a4ad82cce3cc80dca484f1ed39af6a4b58000f",
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
