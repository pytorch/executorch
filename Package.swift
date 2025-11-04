// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251104"
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
    "sha256": "3c89379502993b649ad1e6fd6fb601ae2ba6710a6fc2831fcf54a28466054f11",
    "sha256" + debug_suffix: "b9586b1471c8f5035a24cdd8719245de104bb0e255ab07e160ffaf63ea197e85",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "835a3ad9f6b8945e9f555ba86965ba76984ad19ea048c1058aa8a05a7bf8eb70",
    "sha256" + debug_suffix: "02ed15078113fb6bf9a9d3cc05518672f1b9223edf804873229200aac74daca6",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "78835ca2e1e46977169dd5900877161a403acca0d3c1eec8a79ed66d9d6f6aca",
    "sha256" + debug_suffix: "ba1004eecc473540d93ce5530991c515642dfdbfc68526b7558419dd327273cd",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "addc4adcc2e4a3f139b56c1833011396d8d9b8651e52dfb02410e5b1a352be8e",
    "sha256" + debug_suffix: "a2a261cfe6c66ec1926462cdc1489510decbc25eccf21d4b7168ecffc387ae4a",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "4206db7353ed3f17ef1a73cde6dc5489f319df29d50e2d940426ef31fb3740b0",
    "sha256" + debug_suffix: "e789f524255c2defca6d1a4654e6de6a176e29748954efd13eb6bd1a93c99b7c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "b665587fb3b52984b42362c3155c475470875ed5f795ccc22b0d5190640e5cdf",
    "sha256" + debug_suffix: "673f0b5b0fe9a1fdbf6fcf7c1bddbf88d430c3bb1fad93011b04d1160fcc1b8f",
  ],
  "kernels_optimized": [
    "sha256": "83fb031fc93087c35a5e19aa5f93247018029ab207df50ddb2ab59264934240c",
    "sha256" + debug_suffix: "119466bf87395c6011fce85b990165cb4a7a52728c6bfed8c8e7f4f203d87743",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "075874c11cf6476467648d63ad5d19d01bd58d572e9bc069eb60b8da95908c1e",
    "sha256" + debug_suffix: "00f25e7e91373f1ad28e531c8e1b5c3fdc0366280c0e5de460050a0b8d913906",
  ],
  "kernels_torchao": [
    "sha256": "d4e470891b54097ec748901bbb0686880600266ccb9d10268f53b394fbdb0e78",
    "sha256" + debug_suffix: "6a13e1c64f30f4d313318e77c6ca80e47deca72c37ea975bc670da5ecd1d4d6b",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "50840f16dfa9f34290034d9e67a41dcb4ee77791d9b1d5b0ca908ced594109bd",
    "sha256" + debug_suffix: "d514730ee2ff95f0779790cda687b30dc91510c189c620c2b4fb3f6d382632a1",
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
