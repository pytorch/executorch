// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260209"
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
    "sha256": "7a1440e68fe04d474bda81419d49dbeeaf833b49fddaef0a12672038427436e0",
    "sha256" + debug_suffix: "721df6199854de0a289a0eab97c761f5fd241ba906f032b30376ed63acaf4998",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b82641e216c56ace9b0d52b0e9c5563dae229ae727ea64852c4eeec6f314e30f",
    "sha256" + debug_suffix: "c83d2e12d07cd963134a2673c9029e57099d495f39fac62eb0f73fec1314593b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "65dbf9a4a145d52e2ace75372310fb2e877588e8f8a7de6880382df547bd79ec",
    "sha256" + debug_suffix: "c97c0a0e23004b6239f6c4bcbcf234f125b06f8be4c000eaf262752b4068f177",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5e46474635c17ec1e30c438ef1598780370b5cd98b5d0a654ef2125140e8d858",
    "sha256" + debug_suffix: "92d8001ac4ade812a40d5c73c894565f002e32d59516f7946847c8a86e7c57f3",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "7d52d434a45e0ee983b396cf9d798bad15b00454b798bf51a06a8cd282960c92",
    "sha256" + debug_suffix: "cf1c3af211ec2b67aac06c7272e3392031db5d5c01990a1b7c053af1177d0f84",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "034627c082a4ab44210c361f15fea236f9d3804c47eb629ec89cd127a6ddfdd6",
    "sha256" + debug_suffix: "2640baff7bda448acb4b3a824b549be4cdcc6a2359e379dbdef153ff019f70b6",
  ],
  "kernels_optimized": [
    "sha256": "488c9c176c509f2f040ee18903d3e6bcbf22a9e949591b63c0c0df59772a06fa",
    "sha256" + debug_suffix: "03d169a027a64bc4218844dfc2b22f6037fba4e1f1bd4a85ac71b2dda18b971f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "0e9f465fb671b3ca1108cb65e55c61cf7ea041d3564eb8ab0254f7ee8181610c",
    "sha256" + debug_suffix: "350bdcf6f1ccc1ff8813c3d64f59edf6b3ae98efd5e126844be5fdcd6cdd906f",
  ],
  "kernels_torchao": [
    "sha256": "7cef8cb3e261ce160ad6cfb46b04e2746ba3d5eab0134d2da9299324f645bb9c",
    "sha256" + debug_suffix: "1ae86add6cc4e104b066a64382c8425f1acebb4ae89cc6e777f2e85acd3b28ea",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "fdd3ad13758670bf74f471d57b36c799c513bf6cc4c3216893803fe2451b6b76",
    "sha256" + debug_suffix: "f99c529980907ff05350eb35ef3a9fc7f7bf5de58997b919ca82e3bb0539721a",
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
