// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260712"
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
    "sha256": "5a0a46154e3cf6e73956a27693cfb6a6944498b0c2fa2495a6be10339e9ff423",
    "sha256" + debug_suffix: "35176d17aab7fde14207ae96dee0979e9217ea03e79b0f19f6e943e5d887ba7d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2d5b3d2418cc99fdeb44ac70116731b152a2fa5456302e00e5903e1f4d6ed4f3",
    "sha256" + debug_suffix: "109925aff792668264f395d4e8bfcac681a1eb047707cb0a540ce933235ddfda",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "bfdbf62f1ebf4cc6d4ddaf6261dc49710802b1dcb6198535f2b9dc2e5df896b0",
    "sha256" + debug_suffix: "d63a13088527ef3740f7d90ab48bd7b566bff1c9b5db78bd8c581e44059334bc",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f69b07cfead1cb6ee402e0168d078641c8f023bf1c2cceb28894627980a62749",
    "sha256" + debug_suffix: "7c5861a6d47b86879dee5a1810f580fe76126d30575b1a90bbb85f1b61ca1e64",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e993ab09c7b5ffad3850b4aea71ac1cb696046201f44ac2fea6b6af0783379bb",
    "sha256" + debug_suffix: "137d724a3ab31c0eebb63f93eb65a6c5a446ef7c69ac640614f20309a9192503",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "47409f6ba9c22f0ebff841253e194a182711482e6edd74591a0674d9a56be174",
    "sha256" + debug_suffix: "3be76a5289964312c3c4e6c5d8abeec04d1ddccce9ee6eae9accc65281765328",
  ],
  "kernels_optimized": [
    "sha256": "56f542added9cf8c7e7daf64b282b5ddcc5d5ecefcafe9853203bf2877e27043",
    "sha256" + debug_suffix: "7d7d02927beeca67f6bb505fc0611a716cafc2f921d62b4d97fc8501275c1c6c",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "78a54a26f921a256321d8070a40bf1beb47faa35cc6185cc0c3e12072e31fa85",
    "sha256" + debug_suffix: "04f30c317cc79168820ce907890ded174540881f1ace8c87401341f51d3114fc",
  ],
  "kernels_torchao": [
    "sha256": "4a408118b03e8d09450d7164741c47a55b44d6a45efb7a413ae7fb4eef7eb320",
    "sha256" + debug_suffix: "604487ab05a863a35cbb0139cb9389553e89d6e7d67829b75dd70c9fa59b933a",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "f9dacff36226a753888ae1260d08a8fd1221820315545f49ae5cf250f4317e70",
    "sha256" + debug_suffix: "339d2ec6109a63dab1fdce8deebecd4cb3677118ca3666a34c8e1039776e36d1",
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
