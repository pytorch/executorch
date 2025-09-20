// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250920"
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
    "sha256": "cee1c603821ccaf45dc894128e65839d097fc2046f324286df0ac7579eeeeec5",
    "sha256" + debug_suffix: "6452b9ee8742ca1c5b78388a605311fd843bc84e337385254b1cada33f2be031",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "fcde7ed82c7a0f8065b3c8cbb53316cc3cc0c0042c82ffea78462e42ece492ab",
    "sha256" + debug_suffix: "099c04015bd4dbf4d1cd0df453bfaa95244010e8c081ee986dae4a5a7ab0e7d3",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6d873139402beb87fbf94a85655281cc9ab5ef1739044d55b5045795f1cf0b31",
    "sha256" + debug_suffix: "107bed9d2575d9a6cc2bd5255130155f81a7a65b83a224836138cbbf9971d9b6",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "04d15a53ff304e79ca355d02047b4a2727539bc979cdc3985edcd85398ac6a74",
    "sha256" + debug_suffix: "64e6d57b634ab52bcb7420f64f9f7a6a1ff264c474345a5365d4accc4d2cdacd",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "c890f1cd4815487dc76afcf7febbf240f670ad26a2cfb8a4bd34b09553b2dc86",
    "sha256" + debug_suffix: "8ccdde0dc292bf08eedfed11affae6736f73e48e44ac8504e344e13db2074530",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c8cafff743d2d8b60b1cb330697498d2cf1d8a601a69ac4ce9109768130260e6",
    "sha256" + debug_suffix: "9e577e7a4243073c9a89a4c199df5bc3ba83084b5b35af7941267f2fa24d915d",
  ],
  "kernels_optimized": [
    "sha256": "ac58827fbdf811e1f1bbb124c62ff3fc97adcbde7e8eb9bba77d7be33aa1ef1c",
    "sha256" + debug_suffix: "36da3ada6896e343b913e0e3c089e876af5cb139256949abfd30084164a1577f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "57d8286ed6348f665562513053040ace500fec79f4912c005a25a28d35402507",
    "sha256" + debug_suffix: "455eb2ee380ac51fed2f6e53c82a26762a99da21e7019e115c353bc9273d9124",
  ],
  "kernels_torchao": [
    "sha256": "b1093a6e8ec58b023b1871e82857a3d9ade3e4be981be3f037de6fc5a154bb24",
    "sha256" + debug_suffix: "3aefb0d2e462cd650faefc9b8d60b93e396a998eebbaebd314cc638e388d3cbb",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "f3a5d5a0f3e85042cda603a853d1a0419b93d1c42b8db957bf1ec3be620ae388",
    "sha256" + debug_suffix: "fcc9837e5875579d293dba45e3e02818aba4167fb8cfcc1038e91b7b51dde5a8",
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
