// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260724"
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
    "sha256": "e993285d2b7d5cb908f38e2086858acaa84d8c2cdfa83d9e56751d30e14ad8c9",
    "sha256" + debug_suffix: "63e17298a8b2dfee17b92ed8eacd6554343663a1131d2ca5cc1299795f4f3eeb",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7a50ca5f46957d7c94ebd3c602b5f4359129869dff34f7bc9fcba2f7c324f18a",
    "sha256" + debug_suffix: "1d5bd48eb78421b605fd23ed4d6a7d1fe9f27a8b3c6e5f438ec39422472f7ae5",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ecc9e59e543839a2a6cc6511a0142fd274320a2869f2377d7a976b720573e882",
    "sha256" + debug_suffix: "5c37014517d58a2f8fdcbb3d00de26c2faa9bb8eb48423d35fdb4dfcdbcfce89",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "62f455ab5b6da00eb549e06b959f3622bea59a28d0a678f3a7205d1529a00917",
    "sha256" + debug_suffix: "be7c2d3f2ccbf427e0b3407fbf635ba298f9c3f513c9de58634884c1c045b0f8",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "05e3ee901f702690c76ee7bfd94eee99ab9ed26d4fdc8c626cbf297234ded5dd",
    "sha256" + debug_suffix: "8bf580412cb7a2a49481fa2dae44e1bdfabfe945ec5820a491c81250895b2bd0",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e6037b160f456ffde102c7ddc48e70a5b4c68e75cc879278b558259b81d470ba",
    "sha256" + debug_suffix: "e810277a4415782fcd57d907804c30f0a30c18a7bb84b36aa0b6910e00058246",
  ],
  "kernels_optimized": [
    "sha256": "a22ede76174b2d353b127874c799f72d04a506b3bdbb8574120fe1d911779e77",
    "sha256" + debug_suffix: "a52fc77723433502657e857e127015d05f82aac2e4888615274babeeedb749a1",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c0becfea850ae826edfd819cc7c1882a00ccbbddef4fd121384ae23e091a4050",
    "sha256" + debug_suffix: "323573335550cfe058b204183e66aca7c9a055dfec0f6c03e048271cc2c9fcdd",
  ],
  "kernels_torchao": [
    "sha256": "6b0897776f80710e720b1f5a7358cbb4fc3f3a398c84fc24b550d4404744e52f",
    "sha256" + debug_suffix: "d4ef347c4cd9772037842423b6a207049c96bdab99a94c4a692c186ad4134072",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e88be8137715132faa5cd4c35606c6e6f027e8ad135be73fe2d9c3ca23c8ccbb",
    "sha256" + debug_suffix: "30be92b33fbf854ab1897b6f8911f37f3f3e77956d068be0790655e4604a9089",
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
