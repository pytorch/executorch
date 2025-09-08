// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250908"
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
    "sha256": "8c6154991982e822fe53c18761aec7aa94175c9586e51922638292ed6ba5e905",
    "sha256" + debug_suffix: "e6639c03ff8c8b3f9bdcdc30cf9bd9f4d6e66e339a288db0c813cdb5cbb8220e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0ed8a164b4fef13ae5097c335072a041f0f6409f2fb5dccdb3d226be18c882cf",
    "sha256" + debug_suffix: "2b3c94be266490efb7be7014058ab74fd0836b846d0f56d3c194f78f2a937511",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c9d8a2257ed937ca76ed20132b5bd84debe28b469df54d3acbbac85aeba51a54",
    "sha256" + debug_suffix: "436cf91667cea8869074468347ad73d2971cc46eaef9219c3a8e52168afda91b",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "ae93f45016124651df91a65d9fd5d26b2afdc5ed6d67a4d6353ed88592c229de",
    "sha256" + debug_suffix: "f4ca0ac3c52b3ae51526fe8d33d3a7fb726845ad9514c3ef6445d932dbabf467",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "f75be329e56e2a85554be7bae3cef5d3224689a011ec80692c952c90763ac8e8",
    "sha256" + debug_suffix: "b6a1ab8e31a709cee8510088c27eeb903b4c1cde27eb69addf77ebec2a80916c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "b657649ba7692f3832959401d3f9a01e3326b546efe0cdbc5eaa784f6a932da3",
    "sha256" + debug_suffix: "b148e134dcddfa884f19ee06ba4c15210ed16f151104d1798e49949c293d5295",
  ],
  "kernels_optimized": [
    "sha256": "aeecfde32e326346f5d2498f4709d4eb27087a4531a706daad913d53161881ee",
    "sha256" + debug_suffix: "e2cd59e3934fae6d53f46008f0e29197d7238b8586d9259370bc915940f271ee",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "19ff663e489e0365ab275ff52990b168f26a4c16effa45c2aea8acd1bc8e6963",
    "sha256" + debug_suffix: "58631290984cd7205dae8168b8217ea5df8eb9049ad43258a509f44ca4fe5ff1",
  ],
  "kernels_torchao": [
    "sha256": "e4fac8bdba16cbf3f073cbd49eed28bec7b307c7005fe9e5ba16b81112b463e7",
    "sha256" + debug_suffix: "fdbb0bb8e803a5fcf784b4e7023903fc7749ec391288ddca2154983fd2b889ac",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "0abc708e0eaaa63e86867ffc6e5baa1d82737fa97c88af6249ffe915651d9708",
    "sha256" + debug_suffix: "00f38afef411edc424ab77fb33a78021aaaba06b65612d6607d2910e575ed6aa",
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
