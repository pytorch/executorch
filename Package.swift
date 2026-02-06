// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260206"
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
    "sha256": "850d9368508a5c0d50a8cdc9de6bff739598f3dc0de5715b6e1f917ffd29fbc4",
    "sha256" + debug_suffix: "5ce3babb83c3229468d92eed6492f6c9c13145084344356f121768b503e82552",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "48ff81e31b669f7b6fea5d7ee4da6b4edcb002ff0656ed8e5b05663ccf340c6c",
    "sha256" + debug_suffix: "63bdd0f908e14e5121aae6dbecc50a1bd45ef30f7fc5de374f976eaf6d4a0ae1",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0e0df0d6c3846a656e5a7098eca9ec32b2569f9960dda724e4d3995ca55f9594",
    "sha256" + debug_suffix: "ccc48a86d8572ae653cf2c77dd07d782a0b1c12ad3cda0fa42856ea76789ff95",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d1ef5e1c5bb61a0a8d9f658c09350fdd5d2883276883b6e8b73c1376d6b435d3",
    "sha256" + debug_suffix: "3166cfb21d83db7ef3c133d24e4dc32ec6307c242a429d33e61c73f4c337317f",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "0bc04c6ce6099ad4bc9c88d9f9d85e05e1e105350b5d18ba3f96cb63711c567b",
    "sha256" + debug_suffix: "d09cbd3e48cece0593308846904a88685a8aea2c51cb79ff21945196bc700e5c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "9eda94b17d0c656b24c4edb662c388ad2ab422de689f700d5ec11a23eb30f1fc",
    "sha256" + debug_suffix: "bc04e27713125c332e52f4a73488a4b19615913502852cd8c2161ef94fb804b9",
  ],
  "kernels_optimized": [
    "sha256": "e4c061db099acbc17c003843aeed97395487dab88e380b0f8cf8bfebf285c12a",
    "sha256" + debug_suffix: "46a225e14e86388864fc2b2c0dc293f4e6db3076c251b17dbbc42f7e716dcc12",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "df852871b2bd4ef99644294d5653650f2e2086475636c40ef299c0858521e5b3",
    "sha256" + debug_suffix: "f207ca0fed0dff6c59494ce83bb7b8f8898761e74b7c78ae61e5a7503da3f2b0",
  ],
  "kernels_torchao": [
    "sha256": "ab3ef2d5e38dc9fbc018db4889beb9cb882599e82522e524acd5b0cecaf438d8",
    "sha256" + debug_suffix: "7c870ef37b84be90b429e10d5d50df9a0c55504933a8d7e4974b1baf0800e1e4",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a438d8b052e047f7027837e8b5665d5b814ae1ebcc999010c2a7767668e51efc",
    "sha256" + debug_suffix: "2869ed72e0f3fe7a20e8efd94abfaff752a246e001c9ca307082a1ab36621cea",
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
