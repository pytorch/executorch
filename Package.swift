// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250529"
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
    "sha256": "5198e69b5fe7286903967a8a58e2a952b9cb3f141d31c25ef64e66aa4aba3065",
    "sha256" + debug_suffix: "bdf8df493dc4b3039e2f2a5c17e1db55ff80f775a0bb2e9f5cbe0958452a2217",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "acfcc5049e1bc0c5ed7af4b8fc6f45b8ee5c538b74ada870c9ea9a3fb215fdbe",
    "sha256" + debug_suffix: "05d52ec0e7c362b7de9c3988da067c026333302684fc7ffd77c995ac19d69c6d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "fa99a19977d443fb3bdf92753d6d14b7e6a514e77cf6915220e6a95137b921b6",
    "sha256" + debug_suffix: "4d75eb2be1127fc535b76994fc10f3cca8d4fa4d6ce5f7ec1df8b5f9a71d3c95",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0f7fc4745a5ae6e64c288e494068ed395fd6302302f5b4b610ee3e51e21f45ee",
    "sha256" + debug_suffix: "1c3126f23ddd6445f8622b15070324005a8b2cf73a7542083f9d043141ad91c5",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "0fa0393b4ee4265aad535a41802150a5be84e00527780acfff0a1ae85c0e6bc8",
    "sha256" + debug_suffix: "a6e995f62ec79e3218afa444575fddd2de9165f4b9941999a842c6dbb5072b1b",
  ],
  "kernels_optimized": [
    "sha256": "315cae7b7e848c6bd9a819c47f4878af599dd95f86e352e40ac01299a2106b33",
    "sha256" + debug_suffix: "963a8104f53b8da59de7ce2c84e4e3df636a6fb98d8a7891d0ecbff10e183341",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ea639ae8ca73abd014ef0e27a32f21c586eb63da4c11eefbef3508f57a82efcf",
    "sha256" + debug_suffix: "1c7f96839ba04657787a3cf3307e97b74b661ffbcccf4b7fb74febe3a6cc0fde",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "01a4353d0d28dcc3177e37fc1daf27d3f3036a66950286601bd168009609abcd",
    "sha256" + debug_suffix: "13bc7b0f9f51eaef46ec19dfdda8e84b4979b184468db4a73f72c8e080671df0",
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
    .macOS(.v10_15),
  ],
  products: packageProducts,
  targets: packageTargets
)
