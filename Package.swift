// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260621"
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
    "sha256": "4b5ad0cfc46747b633e7c2ad3926362404401723dae9e47524ef73e99713a9b8",
    "sha256" + debug_suffix: "449a0e721262a61164e931d7b70056c2e7d9fa271c753edd8e79bf2e4b4295c5",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "30d3d347650710086724d43555a4f1b5a95c07e864157a128be0d4f7a37a0f51",
    "sha256" + debug_suffix: "b6db3c7533015eeaa2dbd644a9fa81a3bad979f24c6c829da469fc36db9fe7eb",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "655d4343898211758283846dd63b30e3ff1ba11150adebb686866449afe52262",
    "sha256" + debug_suffix: "988fe925ce729bfba75f32a940ac6addc965ecf023012ce95840e7177f46fec9",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "ea9858af8b3939ac13ddb5c5f2999b6980a66e71ff08de2b9321a522a8a534a8",
    "sha256" + debug_suffix: "3a05a98ba1a2054fbe2548eac086bdb82d71282536ff4b4db1f5253d45cf78ad",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "89a30247a07045150308d62b3fffed0938aac49b4dff3fdd4eb50cee2d2581c5",
    "sha256" + debug_suffix: "6c6a2cf483a27e000ce8bb116ff8c7ab8967cb494a7f2a8b0dd0eacc4305afd1",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "70367bd729ce66f32f92d8d378c7a3f2e0658d72d2c8ba180a4799326931ec3b",
    "sha256" + debug_suffix: "417da6321307aefe61a0722d1d3b21811c36e97ac76e780a85a934fe0a4e33b1",
  ],
  "kernels_optimized": [
    "sha256": "24b9ea02125186f43dfb4a210fa193d2961e420a2fe708a45b845e7dfaf00dfb",
    "sha256" + debug_suffix: "28da5c4092c97375ea4d9aa8372d22a3504d0a4f9d76da0dddd192913cb2c8b9",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ba23a19aff3dae03af2fe9ea2209794b1c5397db617cab451a83e80c865deb14",
    "sha256" + debug_suffix: "357b888dac373a63331579c57a8dce0811a014bd4275b1da6c48caefdb753c63",
  ],
  "kernels_torchao": [
    "sha256": "f44a985f0cc7e9e987f4cdede2235518213858bb8b3f2cb664eabaefbd2f25cf",
    "sha256" + debug_suffix: "faeb18d794d1062c31ee984a7a380c5449859fb9766cba471800e2530d37ebcd",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e55900b8edccadb7f7ab13fa7d90d9fcf1f9fab5e85882eee7aaeca2fd1a3e3e",
    "sha256" + debug_suffix: "b4c0233312916a5e8009b06fa67aed1a6439da16cee6729623e6a95271d8acb9",
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
