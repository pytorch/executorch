// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251030"
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
    "sha256": "aedbdec210221ed42c74ec715cfa219fb98bf1e378fa56d2c53454b1d6da10ac",
    "sha256" + debug_suffix: "c434e866595b91d57a920302fd2c0f443fa3b7b598ab590555d0b55a97f20719",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5f91359dab48790cb907a8e32354433b53c9e89fdc12c3dd6ba201cebb9238c0",
    "sha256" + debug_suffix: "d7edb3ca77c7a40e6e90f32f76d566607527bef696ad3aa15e898d92abf7c8ff",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "203ca0552bd060c5efcd17e353dfba05827bf21343a4587c38f8057c7b7b0481",
    "sha256" + debug_suffix: "f3bcbf69d76d8d1171599632a272e446ea37564a7ed41b7fe6b06cd7aeff2cce",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "29b20dcd7df6ade94599f0683ea75bdfff5ab90f69a38806ebda53e59317fba3",
    "sha256" + debug_suffix: "56e729d50d146768515bc4a07a133932af6e1c51054a25a4b34a1c5ecf890a13",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "6b4e14773b9c51a40a5d2e066456bac526c433adaba4f97aa0938a62546eff4f",
    "sha256" + debug_suffix: "23695f2c6e0576c4d1381ad70f00b5adeeb6edc35e71a17138065d2ea9a3fcaa",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "a311baaedf785ad249206e3a6afe537ec3aa5bc7faaf57c89834a4298d87af79",
    "sha256" + debug_suffix: "0cd08b3989fff8f5373eb1bc5b082ee16151c865e152b2f3bb9fed3ee0f18819",
  ],
  "kernels_optimized": [
    "sha256": "ea8dde81065efbcc4fd14f6f92fce1218d21166721230879c98ee0a30b4c89f0",
    "sha256" + debug_suffix: "048bfbf2b4573eb2b3eb31a1ac49266b3bd613043cccd41d32e124877b9effa2",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9580173d59d80c7dea2db9d90ccbb803822df001af906e3ae5e6520020a4635e",
    "sha256" + debug_suffix: "1be479586995fd03ef1a0dd537dd2baef3e4e5d5b79c98bfade427e0384460db",
  ],
  "kernels_torchao": [
    "sha256": "0e198c3efcba58adcec18408e638563b388ef6905b49f41dc1d7cb79f617dd06",
    "sha256" + debug_suffix: "0342b93a0361711b97488ebd77eef2ba3dd7bf904a0746d1fa912471eac68a08",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a18d317fdca1dfbe2322cb8acb51212bbb37e3d171436c1fb7589742fabb2282",
    "sha256" + debug_suffix: "fa4790d024051880061eab58531592dcc0a1e9a95f5a9df52e164c93836eaa46",
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
