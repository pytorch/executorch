// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260331"
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
    "sha256": "fb006f500db91c686ccf1e787ecc6d734d7faf94764e0c3f70666c4b3b17206d",
    "sha256" + debug_suffix: "9b910f22dcc59b10717fea25e79d2b01816df5dc391923be3d669212f3d2d32d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b7b748d2bc3c3a07396c11a9d74feea8ee906fcb99d2eb8c25cfef711b27aa1c",
    "sha256" + debug_suffix: "216295e4449c3e680b8cd5ec17c9ec6019a5967f9eec34052cba2a087beb7a44",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a18b63afb3a7905c16bc1b29d0f48a6d2636bfc485a7630e7cdb38f48bf79351",
    "sha256" + debug_suffix: "efb73937f43ee6ff041e27efe88e0544bdcd9cf66bd062d622dd12091e015353",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "05f707bdec022599cab5e6b44e8196457c479e83e55dd2c27644b51008d820b3",
    "sha256" + debug_suffix: "322387ad1bcd849135eba6d10311818d7458867f5b802b390a3c6ea69de3f362",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "575a71a2f6659722e6704e3829197e8c30e3c89196dd5a92ed7bc00a5a2741fd",
    "sha256" + debug_suffix: "dd51b96366fdd48a5d02607d00739a2738be080bbddcdc58ad6e556072e8390b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8deb681034303709a544ab99dbe7f1755565d5258fb53b2e973273e9a8ec2299",
    "sha256" + debug_suffix: "06dd3e8a591b36fb5c2dce47bd43fdf3c1def73e67e6f534243d50fdf6185dbb",
  ],
  "kernels_optimized": [
    "sha256": "fd90f1c3201281b60742ff542a2b93b7dae614992eaa50a32d9ff3673b8efee7",
    "sha256" + debug_suffix: "7b1b606667daf849e299171d90568cfda1ff4c26ce5db68db8ac5b6715f379aa",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "98ca3af7ed538b560859f5c606f37e7ec91b297365f854253fa64516c67bfd5e",
    "sha256" + debug_suffix: "d3ddf6ba07d426631c93c72bca15d5964578ab4d29ff6c4d80b0ea378869c3af",
  ],
  "kernels_torchao": [
    "sha256": "539eba90c5e5dca9cf33d6b7226424dbd6e628690102627f37cb6f5913119573",
    "sha256" + debug_suffix: "4b85ba5a5c5e4d4b9c158ad225c7ceb8a3ca3692655d296e78b8d2ac8744571e",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "94ebbb8e0608181b0fd43d27e2279eead13e8cb7a481dea0b47896f003362fb6",
    "sha256" + debug_suffix: "2a102770d49c7e1dfa16504e345936122154df843789434ab4495a9ff8719606",
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
