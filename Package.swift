// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251029"
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
    "sha256": "cefc4c77b069a594cced8d10c2900070ba68ac113166ee68db46383aad357478",
    "sha256" + debug_suffix: "bb9281bcfdadcab6d87c40c1f5d8f8cda40ad46fa8b1af8d9fa04d331c808ee7",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "43e0e7da86fe8fc89974d56690626532e631f15ddd96e434aec461a657c2cb1c",
    "sha256" + debug_suffix: "fc94be4ca9eff783248c5596a9617e0a1f98b233861c5011c38dd4a9f89347ac",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3122a2264522f74fcf4ad83dde4cda65bf7d84afd3166aa547b9e51b0f335ddc",
    "sha256" + debug_suffix: "fa7ef38b95c5a7a5b22ffb834422f8372fe35cf82e1878bbdf482aa5034e94d2",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "4a0d901bb36c656d16849f24fb29d04060c4b881f0758d4cf242edae1a4c18ec",
    "sha256" + debug_suffix: "cec8ce6555b359b674040aba9b645e919729e37bd48780d9efb1e448e28c8a39",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "b1c8f144c4694de636623e75963b35ca18f39dc16df88cd6341cc8acd6de79db",
    "sha256" + debug_suffix: "9d33ea90a6983ae4c47c9c0ad7aec2bcc583c584a319989739ef6d6e30171383",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "68a92d3f778f4c47080b870c6f582f927038b361cc11875e702d32f0d2c5ee51",
    "sha256" + debug_suffix: "77c9b5ff4dee486b0989525318d0b6296b576bd88dfa02161deaed8b8a322caa",
  ],
  "kernels_optimized": [
    "sha256": "0fcd59fbaaf51d0e55033c30fe3f72ba780d66c33bb4ea0d7ff8a06139fba462",
    "sha256" + debug_suffix: "3e70ecbd414061975f407885d141e34236df80e214296768c382052cda881ecc",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f1a0f638302e6abb32e030f2c699e16fd9f699c4e2373fa3ba22b75ebdca062e",
    "sha256" + debug_suffix: "af367166bc33e7bd56f48348e469b4258323d082a2713c86f1162baa2a9fb957",
  ],
  "kernels_torchao": [
    "sha256": "ab94d45a135f606e0a0117e90b1009bb208ea0d5db94d0e37b0c8bfad5f39006",
    "sha256" + debug_suffix: "d1393871c3e300c98514011b2fbd08990dc29952e05d854672c875873c0647a3",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "7ccd5306dc5010ded702e4dace131701348d74e9cdcf0c9fe12252cccef4a264",
    "sha256" + debug_suffix: "5c8afdc9934ce8432b63f65173df9678f6eacc729928f90018b3a7e917751ad9",
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
