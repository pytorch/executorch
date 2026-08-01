// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260801"
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
    "sha256": "0c8d8481f70e80c0595c42fa81e8b5a1f9c9e62d52831dc27f103dcf6d1dfc68",
    "sha256" + debug_suffix: "bfafe3a178fb8304566d4eac7efaca40f06f18ff6842bbbc5b25aef07714e88a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "245f399c071a55c9437164c105f046fc1a79ce3761450ab377545dbc0a17bf85",
    "sha256" + debug_suffix: "c9a9b9bb575d410b87c069137be0577201be9a92ae96067c44737a4dec214132",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "009d1a16005710d647729afeeb6756379cb5b215f9696c3beb4f6a9d662badb9",
    "sha256" + debug_suffix: "ed34573e68a1cbfb2c5a47db4263d4a0c6cdc6bf55887224198a8f7828b90512",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "fbc928eca79ad574380f45c71f25857c9da71d4ac353efbb2f2aed0679564476",
    "sha256" + debug_suffix: "6e94bd2857a9e55dc7718a7dc23789a6e22ca2392df2de14b6fe54a43077339b",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "15caacbaf79bc832bea505adc8769649bc3043ffe7b78464f7013548a3a43a16",
    "sha256" + debug_suffix: "d20e0745c3d609148a292d29d32c3e615948033c1516310fee3c42fbfe3d57fd",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "801816a343548cf6dc52389391870c0905af7f56c726e5e85fae38a9e393d7bc",
    "sha256" + debug_suffix: "232e04019e4ba903619f4cae8a3b70ccc8c2d2593db7f74e96342558a773da9a",
  ],
  "kernels_optimized": [
    "sha256": "0b4481e91e228595afbea92f36bdb2f9c63eaf7b878d7d98cf77ed1efbd72f53",
    "sha256" + debug_suffix: "022279de734c904bd5619041cadc1665155e639c2ebcc49d4f6e56cc872b0159",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "44890ccd95e91a422886331b99836476d3fbefa756fc04ac1a94bcf2c9276a48",
    "sha256" + debug_suffix: "baa723c11ee37f96b36f333c2dd0adfbeb876b1f05c83255c6b8b6b3aa7cb302",
  ],
  "kernels_torchao": [
    "sha256": "08df9578f56bd3a04150d2fc1855be7cb04ee8ebebee139866427c468587b0f3",
    "sha256" + debug_suffix: "6418b986adf8419e500ea933c4984e8cc23c5c376ec7005b6a9496458ef6e4f2",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "6f75df988fe2f7bd000faec4a19ed29389e87cc0cdb4a1d67c86217e3550fe29",
    "sha256" + debug_suffix: "1ad6b3dfce89a0360ecd866a7f64afb81eede6cf8e6139d391c169dadbb5e4e4",
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
