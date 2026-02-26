// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260226"
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
    "sha256": "e84a3cbbea40fcf9d1fb3ae5802add4b3d9c6c05576755c11f157984733d821a",
    "sha256" + debug_suffix: "5d09e2bc77139cf084a2a4284c6e5e8051582d126e0a5b4ab36e90f4073e6e9b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3f9af85d5c17d8a2e076561d70bf2eaedbdd00687f39a113d97aaab42890d715",
    "sha256" + debug_suffix: "0e76d9ab4027db861e0dbc4837cd2315c7361543a9ce176beac46ab5a93fc894",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f6eff6466c9fa022290509b73989510c6ee1c9caba410dcca05cf6be58b1c1e3",
    "sha256" + debug_suffix: "aa3000eee1b25c07f50de9a73f559be41caaee6b2dd4c53a92168165e2c82195",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e41e2540f3b536c0396cacc49403bbe5f093ecac32a5294303e886135d5a9db8",
    "sha256" + debug_suffix: "b7fdeb379f15ca7c2d72d28931da462891cd43486ff3c92f10217ccdb95ead1c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "36f3b59ed2eadf932abe40ed937197b27fa7e48835eef687940b6987930a0ab3",
    "sha256" + debug_suffix: "c9fa94b453ea23acee3ee92073cd96bb53cf856ec04e5193e8d4751f51c68094",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "b91c8b7ef3c410f0edabd3d4b6ab77fc71b7d31edf5e441ba19899733f9b27e9",
    "sha256" + debug_suffix: "61c4f88cdd096678b46a48b1cbb4fa965a2061344bf884f992db91f4df1fa84e",
  ],
  "kernels_optimized": [
    "sha256": "5a41f5b9cfb855376c796191ad88a77374ff11b6c24d2f320c8eb2fea7ba49dd",
    "sha256" + debug_suffix: "fd5deb258569d0e9bf3079bb0db7c94359ea9b235f51353481e7b21ed6d25a45",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "765b5af601f20278b88797e638fba939117d69ff3cbf2a433639c08147db2d53",
    "sha256" + debug_suffix: "2a83f38d7be166395782bf0ca61f752d66aa5ab191c367b25a9989c0ba269954",
  ],
  "kernels_torchao": [
    "sha256": "8d6cb7c9c7467e4c6885d3ab411ae7e162168cf8e6a329cbed0684a03deed486",
    "sha256" + debug_suffix: "6f7fc1489bc4f2597ac9cf9bdbb19b58297d78c4016f5c5d66e28f1cb007f6e7",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a80b3e73f0e6311f157153b1f073ee3a2846bb89ff19527897aa24063d07f92d",
    "sha256" + debug_suffix: "399456990c9ed7e6b0fd880f88631ffda72e0599dbf6170e97534fed4695e717",
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
