// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250917"
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
    "sha256": "d2977d11a8f30fe393b5fb5282c2f377b0cbfc2ba729d7bf6912a175df957bbd",
    "sha256" + debug_suffix: "c40c174b935aa0aebf1aa83a6e3e8d3d9f2b5539260e2e686c3c64e393fb8047",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8a2e968c2968107392f6cc3232800b37c064065f6a5bd4f7799ac125891a1115",
    "sha256" + debug_suffix: "4c4e9f514d1d7027c6feda15385927b8db8222ecf9ae18ca6e6b32e55bfe0915",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e1e8eb2fcdaf86d11d725a770059e8064652280273baf000ebb2454419d77833",
    "sha256" + debug_suffix: "715777579f336e4a56b40a44ee7cbfee5e45c160cb67e841f43bc71ffa0c6b04",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "7feef37e3a31870e5d903beb7a9f1d1767886955690ec2fac7cc58ca81188e3e",
    "sha256" + debug_suffix: "ab291e9a62be6f2bd00190fd8c4a2277405fce18d61c0d198af085a5996d57d2",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "10ccb112b7294b4d03347e31f5b6cee9f0c8285bef0ee82048094eade66ad30a",
    "sha256" + debug_suffix: "5ad8a6ee2def189c3400e28a24d6218df8b49869cc8bb7dc3f7a9b96bbc1512b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "5c02c967a67ef18d2b2f33e1f66f0dd361d9db229cc7c4b38660f68fd77ff66b",
    "sha256" + debug_suffix: "9ad6adee64d1dc17d893205b4aa9008ecdf35b7b873e7b1e8af9342cd84eb039",
  ],
  "kernels_optimized": [
    "sha256": "16aebccda2897d5fc4279eb58150f61040e64b1b86cab817d92df3babb1dadd9",
    "sha256" + debug_suffix: "e4bd77e6c660b780dd71a04815fc414304c84aac7a4f7d0d53c088ef1a1d8689",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ad8d76583e35e3f1fdd63f4a903fb3a7a35df71c5cbe5c701b210af7a3c45bfd",
    "sha256" + debug_suffix: "d55a8b09b2295b817957c69d486cac3513aeca216704f241490f70685f78b744",
  ],
  "kernels_torchao": [
    "sha256": "554b34c8d2398143c1de1fc6cbe43e71ed4bb5441b59a2cc40305dbe5aa98573",
    "sha256" + debug_suffix: "cb7c710093a570c34f3f71320fd47288d64df2f654a9f87bc6c0be6f0045ba83",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "52da4bd84758a871b8dfc1735d18e4eb45ecfb5b7cb6b1324bc91b452d500bfe",
    "sha256" + debug_suffix: "f9165e980b7c926a1ba840bb585e1601abfb3ec804e71db7e91b5c69ac8eadc0",
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
