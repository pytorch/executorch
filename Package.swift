// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250705"
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
    "sha256": "b56ba02b912ea32daf75d5ac42266fa99ea963e3fe1cdf8e81b6be30b0e2ee13",
    "sha256" + debug_suffix: "1a90df7733e4addc242583c41977ccb96df8f9955534f57937e35312d5bcb0c1",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "34bd1e3b64bbef7acbbcba410bc71c866331977b69e4937b4ff30fbe6ad769cf",
    "sha256" + debug_suffix: "9847f074f8aa4ad037124b19c86b1e5c22d95fdbf4529bc0c7e5c935e9ff6c3e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "bc5c47c3428538e2f9b772e1f8db74f3ce9b9ce4206c94a6f4d0266557c27526",
    "sha256" + debug_suffix: "4d2b20b623100efd204c9e770f45a21191e9a11e8bf750250e8557b22f7948d2",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e4e186a453645d0979b6e2eaa2e52eb9aa5bb4de63aa63a4afba83b375dbe9ef",
    "sha256" + debug_suffix: "ca488c58e7edaf0e0aafc340f8c6b76976ab33639abec458c27e33f738a4abda",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "d07b245c2ab0cab45c3eb138979f4e899803632893d28c7599787979fe745cc6",
    "sha256" + debug_suffix: "778bbf53527588e738a37b8158f4fb91025e014da1d5ebc718695a97212d5cb4",
  ],
  "kernels_optimized": [
    "sha256": "88d554df074435b2d3581d917b45eee249489a1cb040ff95ef8dc1279f21aa13",
    "sha256" + debug_suffix: "b4a46727bb7b1484963088ab32411bb29c3a1db286d0e10d18dca1e87a18c2b1",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "6db92306510d98178c95e619369771df6e665903409da5cdd0a86ccd85eaf2cd",
    "sha256" + debug_suffix: "5689a5e1fd8002e5dabce67cb74ba712e0939d0a9000fd3005df86e58ea4e031",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "159a9a90a1cbfa310fe6cb3a5744b015fb9ae03d9fc572e809da58a8b783a53a",
    "sha256" + debug_suffix: "8da93a062641a38fdced03d30fa4318b8ab0239f10c0b5c189162912390ec57b",
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
