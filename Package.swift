// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260825"
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
    "sha256": "27871c981699046845ec2471bba8ea81dd5a7bfc8c2af73d158c06a614a3d057",
    "sha256" + debug_suffix: "499ac9a495aab5974cc5c532bef362d625e1e5abf60fbf4ed0b77a5d219410ad",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "553950789f31acde342c2c23523133fd7a3236ce1483bc6aa970813b809804e0",
    "sha256" + debug_suffix: "7e33fbd06e5b1ff57a2dd2a420c4ef3fea3653c6ad980f5f752ed5c199b750c1",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1a1bbac21db7be702693fd6bd4fb3713e650c1a045f78aeebaba43c33f6e1694",
    "sha256" + debug_suffix: "aacdaffb8ebcdec05f21c1cb5c21ece7dcf5acf111dfc4c4e4f94534a45d8f30",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "7d033b20b79e9400f88bc37c5e3126dd907d577ec65ea55e4be771c9c24914a9",
    "sha256" + debug_suffix: "a8e2457d615f7015f6850e78db45a6b32f3e596e094e8730b0ebbfdd1ca797d2",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "34c7b7dbca44af19f7da4cef1c810e6d65893d5f2279fd0d5792f4bbd13a7e86",
    "sha256" + debug_suffix: "de4c3bee1b472c97a318c0bef317640e2cdffccb6e493f14154a3cc715cd60d8",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "9e1215dfafb730a6f288198ca4abb7a4a2d0fd112f67f239d91a27a805aad00d",
    "sha256" + debug_suffix: "c75dec86776fd55156196ad15b856db504ebf052a8385dc769a85e6252dd349f",
  ],
  "kernels_optimized": [
    "sha256": "ba132386bf432f8dd1ed39edf7990f53aa0b4f9425cb8c7265b419b98416a693",
    "sha256" + debug_suffix: "fa30ff2c5b63ff523aec7e466c85f5c06948a345c9dc29d423f6360628822277",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "97552e23a210e68e05788394da4d9712171e34cc179e85b791a88e2dc87491ff",
    "sha256" + debug_suffix: "8dd94d498814aaa2f8860978871a19a9de08d49d613164de9486bbd22b86a6e6",
  ],
  "kernels_torchao": [
    "sha256": "16d42b1a2e79e37c47f759128c066a5520f3780d9090fab2dcd04a56960336d6",
    "sha256" + debug_suffix: "c08f169a69a7e3251a37d1d93f66e1fe812df07abb1ddf4dfddc3f61589bf869",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "d96dde5bc064a5bc4da54086193268f30125d2d7df1714ec32dbddbd239312d5",
    "sha256" + debug_suffix: "20179a2409b4d11e341f189edbaad088ea40725f976df1ddbba2e8f8298098c0",
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
