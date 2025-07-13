// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250713"
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
    "sha256": "1c71aea9107cf90a8bcdbfa56ec90d98cd0431c0e52386a0c4c2264c27aa098b",
    "sha256" + debug_suffix: "1349a6ec845ad208354cc071e0899e42e350b103330a1950007ce6d4e4dc65d7",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8ce645d655d4345c37f86efc4188b3ec391d811da952598b631d2bfd4e4d70ee",
    "sha256" + debug_suffix: "f2126ca61ae3e7790ec849bb93c4cab553c10076b05278f47384b22eb815ee06",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3349ea38d3625c8875259d0f9622ac8e0e97791c604f79f903f4e462e4eb0807",
    "sha256" + debug_suffix: "d66fc6905c8c74098f8658a53c62ca4847d8f704750e476831963c9cbc805b07",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "04450c5d88ade612b8b5b5bcd67b05c4290d167b4c2f9c11c1145b65b74a316b",
    "sha256" + debug_suffix: "ba95daafd4ef7702d935b324725b6dbd9b0bb81fc97b66215f0e385c57675525",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_llm": [
    "sha256": "9ec6b00125883ad98648ee301fbcae583938b6ec76a296d4e207a5810c3f0118",
    "sha256" + debug_suffix: "2f644893c142af9a4b267590be7934f7c1c2897150990e8971c95eacdcab247b",
  ],
  "kernels_optimized": [
    "sha256": "d4814f158754817156fd5723f2ddc0efaeb2069d1373265390e0b76cbc8a3d00",
    "sha256" + debug_suffix: "6ffd6c907c4499a146d5cd6b1d21174dfb478f43a263dfc08f7e4f5d7437ce24",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "0520efc6dde97fccf789e634b8a3525ab21a9cd1adedea2dfa6be5a314e3520e",
    "sha256" + debug_suffix: "87051f2fbaf68d90962ac9b59e3981a7b90cdd39451305dc96bd7d1a4719bf67",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "58ad0a6254137130da0a16fdd882965a54e2a8329ed91fa989cc9ca69581cdfd",
    "sha256" + debug_suffix: "28676300bbb7009071bf653a91b931c2a9a08113bfb8d9cc4227807d6e3f5501",
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
