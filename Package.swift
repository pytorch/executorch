// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250921"
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
    "sha256": "59b7dd8ce65eb23d6f443154f041a8ced5d46382cf01684ad14e37cf5a548f84",
    "sha256" + debug_suffix: "ef541f2d831c4deacb3ea79b3896b13ae347da75ad992ff7af40414509407f79",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "fee005e6a8b5340763a17ae3c1090cb6929442401d60033acadc36fede47138b",
    "sha256" + debug_suffix: "21c43f60662b3568cddb29c5fff3c0ca19fa491697402a353790fbe060f8b223",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f69188c06930e439825b84ec7fa508002ac5884ef98ef234ec76c2eeb26b86e6",
    "sha256" + debug_suffix: "c67e183f3ff794fa93a2feb799384a2efa6abccc277342452ee84d2bf0610432",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "17254b44cb0ca1fb9cd6cba198a75fdec92e3b520febeb78820a5b137ac854c6",
    "sha256" + debug_suffix: "d3684c663072f255585aca2393030a253d524c8636b0fb1b97e2f1512b10385d",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "f9a5f6c35febf510738396c105bf4b8c1b3252b7561d99ba63e902075b04686b",
    "sha256" + debug_suffix: "2a170e69a03973828197a37d7837cb2a0a1c898202d8a9ca6603546edd90e72c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "599f34ff72d76322eda7f8e4b23328c8772b4f3917d75bf3300cb2c0063818fb",
    "sha256" + debug_suffix: "060151d16a659105e41f32f3eb456752aff5e395b9c82f1f85436d709594a89e",
  ],
  "kernels_optimized": [
    "sha256": "48fa8c13f112795c6ea7b6cb8b6aa62fba1ce5b54f4af65bba7c3c4374c7c1ca",
    "sha256" + debug_suffix: "650f6dbb2987c3d980c60e8f780896ee0de95f97b80550ad6c95b5b23945d119",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "179017985c24d1fde8a0c84ca750c0b9fcd34da4ceaba961bd8827f2d9c9e699",
    "sha256" + debug_suffix: "adaea87a36e3c0e3ece8373e19891025663736bb5aec9ced9132b777b7991f92",
  ],
  "kernels_torchao": [
    "sha256": "de5082ef593e69cf3a8d24b2c10ad97388bc5ed291be6a7e4e4134d85a1f88bf",
    "sha256" + debug_suffix: "70d2a0efef4c2df9490367f7e8dd87268af284c1b35cad73bdedab032a75c3f5",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "2524687edf3eaf515dab8ddbe0ae9b828d8ee1fcc46789e61aaba4185b7546c0",
    "sha256" + debug_suffix: "ac09bd15a7100dba9fbb3745e0b2c5c65245e131af08f63f7c92f45cf0acc3bf",
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
