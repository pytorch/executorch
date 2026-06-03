// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260603"
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
    "sha256": "148a0d9c217d9ab73498a3226016a63d29abf18dc1729b0c0366ac91265cf510",
    "sha256" + debug_suffix: "2f45b756d9a85511f60526427b9962e70fa28ad0b54ea397a79a0a0b85929041",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "fc6a9d1e72a3c9d76d92f78516149dfc7cee27a3b77cce6d22e19cd0602af065",
    "sha256" + debug_suffix: "1857d0269a823b6b2420ec629a2d66fe307294eb281ecd96d8e9ebc7b5f90145",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ecbde3e740faca8bd1c1d23b2b09084826a1d39b127a98cc52b8d84bec4eee7e",
    "sha256" + debug_suffix: "9d8e8dd0e01004e382ddce2ce6e826b612a723681ee8bba37638a007afa5a3d1",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "4eeeecafed739d5c44e559b4a847dbf38fbd650e8b612392e77c5d970a1233cf",
    "sha256" + debug_suffix: "39a33c95fcd5c4fce55a1090fa9559e03742c91e56b3fc28f229828dfa75a150",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "90b0670ed5eb1a248066fb594192a20197f207526d435cfa8872e689ace1dfd7",
    "sha256" + debug_suffix: "9c9b62c10997d462d2c736c1b2e82cebccafa390a20c6fffb56d1be2c6d33c2c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "762377b6e46e92f0d0d1a6e0e9360fc7c6d1233477545c1312de908da5e5cc1b",
    "sha256" + debug_suffix: "5385256f4dfed4dcc19478731bd75e736da5574f4eb6b9ccf7cf21e2aae62bfc",
  ],
  "kernels_optimized": [
    "sha256": "b570b2e28239718937d9b8475119140112449cea48d89c2b59633c908e245d33",
    "sha256" + debug_suffix: "1953133a3a2e94243e6eb291d480c2ec53f8a03f71888e7c70e94a892744603b",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d8a4604c50f222550d7746a248d61dd5d010e4091173c746fa308b5b7b5e5531",
    "sha256" + debug_suffix: "860ec4401f673d692af0e189495f3109ee63179edd41a6125fb048c714a862d0",
  ],
  "kernels_torchao": [
    "sha256": "1e3d5f1fd89a8743572384233ee1fb6b36e81956e7ec043f7b7485cfcc45d93c",
    "sha256" + debug_suffix: "670485af26bfa66a929572d1032c0581eeb3bc2dff6d21cfd8d8d26d0c2ad806",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "77c6c4344b1297dc7ed296da45e3c6d98da81f5a16e3ad635d76f988cd90c246",
    "sha256" + debug_suffix: "7ab92b650485147d4c38afc131fbe1e33191488eeb7ec41e49b73a98083b9e99",
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
