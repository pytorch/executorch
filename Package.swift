// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260610"
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
    "sha256": "47c7a32a547a22263487bd7dc36b0eeb894e115823f93874bd36479e509f243d",
    "sha256" + debug_suffix: "a1634a255b59e44a0853c9e6a0adfe269be24d1e73cf2947dba7fcd0231894d8",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "c90755678ac1e03e5574195315bf47b6793c821a7ceb3417905f7885204cbd83",
    "sha256" + debug_suffix: "40fecd974cb16862bbdad484119c091625467130c6bbdbe0727deb6662b82e38",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "021f289dd198fa6fd6595ef71b6e660259450179db959c072f4f9bde30c57be3",
    "sha256" + debug_suffix: "098d74887cd9b9cc38d5c1d3bc982c2ca2b546602b570d13e04578a27903b81c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9e67152995764af3f0c5372de60382301ed604df66a4c2e891a1d0e1488fd8a0",
    "sha256" + debug_suffix: "2d5c0734f7f4edf9d15595eee8ec2f74de1e50e6e2f583ab65ab657727f876c5",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "c03883b35321c19bd346bc927326c31e1cf66c33049fea90df60b374ea3c2881",
    "sha256" + debug_suffix: "a68737cca80b8a27fe2fde114b069c906b0e87a6cfd7a33269fa526a34b90031",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "aa4435833b814061f9775265fc3bf324382c30a9b14b8ed657b740f55701d067",
    "sha256" + debug_suffix: "e9c847f8a78dab8391ee00cdea0326a9df2f520f525018274dcb83802ba31139",
  ],
  "kernels_optimized": [
    "sha256": "591ce9bb47509224227976f2332ac7a3293bd38a63f1ee4f020e5e557323f8df",
    "sha256" + debug_suffix: "ddad0171a2af2a6a9aacc633a7aa94e835978ba4ace9b62026d9baf47acd797f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "91dc5004a20f0754b08c4a42acec8862e7bce7a16a6e20972fbd0a71a29e9158",
    "sha256" + debug_suffix: "6957b17390cafb69ec0c5f8e6b6cb9a56c0d9877d1cd6e42587358d46860d773",
  ],
  "kernels_torchao": [
    "sha256": "9921ae9660e1bda4ce2a79004f83490f94bdd36101e6f647af2da9d7495cc02c",
    "sha256" + debug_suffix: "f08e63380c620f5f7496dda41ed3f73d2357cb691ea226c236885cf26bd71285",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "17a80a4034e0ba16169ba9d97e1c15334e90495113365dc56fa5fdb46b054d67",
    "sha256" + debug_suffix: "629f73c18cdd347d68481a59c5a945b60b3f18e9f9ab18481ef2f5b48cbfe0d1",
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
