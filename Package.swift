// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251009"
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
    "sha256": "d844f6f643a06433300c79f39561b3e1bc06fc9f05db72e92f0b8c336546c2a6",
    "sha256" + debug_suffix: "b4db98d87f532d089fa487f61ba36e0dbd3df77de975f758dd7682696630feeb",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "fb0265ea064c04ccadf2e73aa59b56b0ebb40ed33f3859fea6dc6c170e86db0d",
    "sha256" + debug_suffix: "5ec68c170e721183ffbd82a576a843a4508bd56d818334637617ac97d0882592",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "bfd01ceca85ed7041490ecb36e482c9a410324a6d5230d85937c2ba3782b01b8",
    "sha256" + debug_suffix: "07420c76deabb36054f17bf2df543cba3b2cf02964cb99502284bb177d7b76be",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "aa2c3e6499df042ae674e9c6e00303d5ec87be6eebf4196b4466a91072c7ad23",
    "sha256" + debug_suffix: "45168c69162883975552d890aa77cbf6eaf6d8a3608b62297ffad322f5d3e351",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "b91a1f9379a5fcfebb24319ea0dca84431ae5f2ae1e26eb2d37ec61d801bbae0",
    "sha256" + debug_suffix: "0ce81104c4c04840912ee16a20d92ce80d9fbcc98ae8e32696fd0668ad6283eb",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "d917df2d641098679e6ab2da2e93ead3ea3e3608291cae0fc70c3e6de557b9da",
    "sha256" + debug_suffix: "528dccf58835091457cb4df7b53d7ad10e56054c3d23b2f733102dc3e2bf7a2b",
  ],
  "kernels_optimized": [
    "sha256": "3db5736c5ebc52bf95c6a643c9eca6255169f30aa80a94eff70a7f17e567372b",
    "sha256" + debug_suffix: "ebae1454b2c56d3ee7569e46198bd75921141ca4e749be8a87d5ae8e7e0f8e7e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ec514c3de1892d036d61fa290a702611198ff2db28850699e1128031c2c0a83f",
    "sha256" + debug_suffix: "e1d15c38e020f62c18c9927df994ad8b0c09ca46562540cdb3faa043250c2f26",
  ],
  "kernels_torchao": [
    "sha256": "9409b801af3c9a18f4b955c97ccfd8c0fa51c9a66985ddfa7000bcf920f4e72a",
    "sha256" + debug_suffix: "b334f10942ff9d16f2f07353e9ba68c5e8eb973656238c4afd5ed7c8fb196d17",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a8f14c6afac637ae6149c314774c6366b5a1e6a153b6e3846698a4ec1d3140d0",
    "sha256" + debug_suffix: "0ffa817fafe86fab142b0a1fbf77bc31a8f68eee3562574a77a9d9932971e3fc",
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
