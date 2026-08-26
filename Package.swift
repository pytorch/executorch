// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260826"
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
    "sha256": "11ce225ab9ba856c058e57e28d06939d44ee4e98d09588db322312dbdf05004d",
    "sha256" + debug_suffix: "4d793462f0e2fc93fef89f5ab39be17f155f56af139abb0729134f95e2c5072c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ccd0ecafa767811df76e35120a6e2d00890e044b9d45f54bca11bc62709176ea",
    "sha256" + debug_suffix: "c3c8216b859b4fd36c0ad79907aa2e009313fac8ae97bc0ecd39842ff365c490",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "65891e11d00fed41475ccc4cd38cc73b4a59b3356735424d035638b2175abcff",
    "sha256" + debug_suffix: "12708d62782d2f5f5256de78106e0cde3089f844a0b1b56362da72c659cbe098",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d34cf836587bd274cc559ca82a9bdf487a2a8273356e85765eeb54edf428d88a",
    "sha256" + debug_suffix: "b82db8aacd630af00355e117629f3495f1f562a09c609e31fa56ece63827922d",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "491995bffd4d6f0e60b394fc8f9fc44603ec5e23dc2a0ec4157bf94fc453676f",
    "sha256" + debug_suffix: "ceb2949aaabf4b5b69266e8e0a3394856f233c871cf2ec3859479204106ff534",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "d5358da0eb044761996550b433690388cc38dd451313aa5e9248e1aadfd33e09",
    "sha256" + debug_suffix: "40761d488fccf288a9653c7d6f0f11caea167f26c2d47a07dc15e63627f29cc6",
  ],
  "kernels_optimized": [
    "sha256": "8afa74d34c9c945b576968d15086773422b63b38a8e519be2b2d3be1c4e69e60",
    "sha256" + debug_suffix: "4f2f89a11b54cef390dd9d7368f84f9d7b4c325fe7b6df723d050bab20aa5fa9",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "837e28540f8c785416928121fccc76db550807304ebf899ef180aee813225e0f",
    "sha256" + debug_suffix: "94603f0f43085d1ae512be7bd8b555f2d26441af3d03334b4dacd4ae8a6a3e1f",
  ],
  "kernels_torchao": [
    "sha256": "7812766aad0de70a59bbbfe0c809f32216fce836ab806e2f613071a180608524",
    "sha256" + debug_suffix: "9eb0be26c5f8c91b8cca17c4781728685c101abf12b3e1d9e3cf361d8fa445d2",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "f60d5ad79faad530b839c866f9661c74f00c472c9117dea01a36fe56b93d4a11",
    "sha256" + debug_suffix: "1b4eb3ebaefa345e846c2afcf3a08494003accf8937354416dcdebb303e74d6d",
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
