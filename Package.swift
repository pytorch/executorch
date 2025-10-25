// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251025"
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
    "sha256": "d2d0a373e3d3f9c821ad5054f6658dcca7dbc14bc924744edbf6341594758118",
    "sha256" + debug_suffix: "ff9b45ad516c13e87f1c267ac40fcc2cddb935ced784d541fefbcea0849946f1",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "12edee980329cfc30d3a00ff9a7f2b599c8f91ac1d50e021ac2075186c75fd7e",
    "sha256" + debug_suffix: "743cf9ff83fab8e3544a677c8b9661665341aa97b95a36510486c2a29847505b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3191d917c97e618715b106d4a6f102a5b9814129090ccd5ea24bfb4d35e0565d",
    "sha256" + debug_suffix: "251a571d6947a163e348954ecefc3472182bc97058a9adda3db773533cee390b",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "05eda407f25d4603adbb3bcbd0812f97bbf75b027a756e194a7a079e3d14fe75",
    "sha256" + debug_suffix: "984acaf4b132403a6c74de3b0a152b32788290acdf9d0901c89f9b4a2cd6d187",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "5defaa433686351e37f1e51f6c25196f19ac714c6bb84166cec67d560333b238",
    "sha256" + debug_suffix: "2dd70f7f730deca517d99b935c3a490e571415d227932ba51369eee3748dba45",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "1feaa646ab266cf110d88a4fb65ed39a02a6b9d00551561c77bed2a09dd4dff1",
    "sha256" + debug_suffix: "e9c561c299a941132a3120ba6d99a899cd41507ccc77bf1a86c7d2070c9d860b",
  ],
  "kernels_optimized": [
    "sha256": "2f0a5d8edfd20d69c913157be5f885c7c4ced41553335d0f75f03ecf57808b8d",
    "sha256" + debug_suffix: "4d725622d1aea831cbc050c572a867972a2ad0769320e29107046353a0f2e64a",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "26889bed26148663acbfe19e5852c0db3221e7d3b9811bf76719cec4c91ad282",
    "sha256" + debug_suffix: "2f8cf4ed8a778462703d319cb414e45c223b5c0cfa8e35bdd5f7a9dad4567ad7",
  ],
  "kernels_torchao": [
    "sha256": "9f2963750d448187c42d25e75852ae5a7fb185cad75f9e2d3054488a100a014a",
    "sha256" + debug_suffix: "a4f3d85fd6060c428baae9fbd76f898f734a2ead0ed1869e3697bf2da2640a3d",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a1f5fec45c88552bace5684fa845749952db3772dc3998d5660b9327f41aa269",
    "sha256" + debug_suffix: "bc0a813e6748e0cd5b0101551b97790ea9654dc3ada2a6e6c991bd4bb60e3c9b",
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
