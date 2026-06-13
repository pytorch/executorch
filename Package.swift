// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260613"
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
    "sha256": "de10801d9e680d35f3f47d76e9a6c4575c423ddfe423eda18edf1d0ae1294b3d",
    "sha256" + debug_suffix: "a034d86e5716ea5ea0304a92840135916de62bd3648233c96b329aab39769d9b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6394a27db191bbf6d1c579aee209b511173bf8d988b8e1c3ef3e771f235ecd4a",
    "sha256" + debug_suffix: "8697e87dc95f6efb7f3543b87619f9baf7ab155a0292242f0f6fdd24061acefb",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "bc9365ebac810e458a7ac88e22a059f070982970716fe86322d5c6762c8e565a",
    "sha256" + debug_suffix: "40b02ec2bc04de90b7a48a5c27766f28ac6b9769c9286f1b1ac4ad15b38a79c3",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "2be4a480c181d95ba2da4cddaf699f21c29489400dc573da576bd218ce85edf5",
    "sha256" + debug_suffix: "d2be67e18812b8fe23b3446144e145726ed5a3d01c16f49440d19770156acf4a",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e7f41e059eb21249327c5330235fe324bbea71c9920c616c64d6860eab144caf",
    "sha256" + debug_suffix: "600dce0f60a4daebb86a4becf1d6ff8b72c542d2588228cbc15516c2e13398da",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "6872d640f7fa1e0441329fe0f11cc8371553edbab92a6a514ce402e42f9e47f9",
    "sha256" + debug_suffix: "08c797211fef016253e584c100af2e2f81fb7f6d320abf6e32f4e403ef02d6cc",
  ],
  "kernels_optimized": [
    "sha256": "f4db76058b514f871cd5a90253e8232bbb2126bb8c13c45654680b0c03d6df4c",
    "sha256" + debug_suffix: "4360a5934df9023251331f04e05361fa02771c0a16ce620412811d54aa8f7101",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "2c7cf6ee5f8954e292aa54bc2ee5d80dd1132a76d3066f29a04ba98f7e1701e0",
    "sha256" + debug_suffix: "22cd3f6c83cb8a3a1ee2b8705e40a2ed4967d4838de6a7c7e4eea55aaf88afd0",
  ],
  "kernels_torchao": [
    "sha256": "d48c88281cbae4304c826dc70244c2d319a96e9f63583dc2c7564cdad7fb2095",
    "sha256" + debug_suffix: "800307ec8bf05c083c7290b0c38d9d60ae2c54a608edf7c91102e539b1739612",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "3a94419bb02b3f5aa4ca2b897f414fee2299dd24ee2cbcc529c7909911117772",
    "sha256" + debug_suffix: "5fc6a17a5c2c40590b187f41f98b4974e57ac895ef856abc811a6823fe13b52d",
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
