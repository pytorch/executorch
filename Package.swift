// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250701"
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
    "sha256": "e759b20b1468e0ca779414aa560f99acfdab1eac6acd0bc0f3e9110f5d760bd5",
    "sha256" + debug_suffix: "2aa2671efa4c89bd58467e4f37084c81aa6390a8745bc5996f3a50bf27cf0cef",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ddbc27cb1a83c8757ff6c277ab24653d31c1f85ebf4c91bad33f02e87f29afa1",
    "sha256" + debug_suffix: "174092504c68a8615fad913616dc3b5d48c96ca3694f3c3b38b3f01330bc7443",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "72dc970ee00d7040caaded7fe16208395ac61c97597ab498c7534aa8ca8355c3",
    "sha256" + debug_suffix: "d2bd203834142ed74d57c111a737f473228af1734203fc9b8b43bfcdedd51a1b",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "8fb0fa6abda4c3b5cdeb52ab5e5acb9211cf2465a1ae98dc574edadebc1aa814",
    "sha256" + debug_suffix: "52442e1b3a91c93edf8a19c50c22d2493c7afe757e69349937c91890a39e0af7",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "4c9028eb4734015d699c34adada16df252751dfa30291748c5256612b4091912",
    "sha256" + debug_suffix: "ad859cf896acfca88f1b6dbc6d2f7554d85a5a3b502ebee7060a7b86e3c8e781",
  ],
  "kernels_optimized": [
    "sha256": "f1cac9b21a4d9e95c92e04fd44a9a7e5e741973ebb9d068425c722a8c4cab020",
    "sha256" + debug_suffix: "13714dcccdf803fa50a68209906dce9bc1a62c167a4e59572b2d8f3daa006fe0",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "7f60b5219afe425eabf4d6721f0a6466c16d42edf042ec09a78a179b6ad9b3ea",
    "sha256" + debug_suffix: "cd7cf988b9c8f4d87cd8de6c964e8441d0666d4a66c67648c06ebbf427fea24a",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "de0fd87aa02e5531723e6311a693f7a7e85d62d576d669dc3543e821dd6a8712",
    "sha256" + debug_suffix: "dd732acf322880f7a8227bb0d09e8f78b59b198ee014f3b9bedc8a97a518b199",
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
