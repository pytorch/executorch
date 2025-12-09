// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251209"
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
    "sha256": "280b4816e7da64eedf3316fc33e21aad1dc96353dfcab92cfe690294ffea4e49",
    "sha256" + debug_suffix: "ffe9b84c3b0bf6e11d7ac2969fb4da3e9b9560e1a555c841703890791b5d551d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "dfaf46e4d5b10e880a2b8335212978301e62fcfb76cc0070c6bf97a1412ef60a",
    "sha256" + debug_suffix: "278232e4868db87e5495743d28d9cec54083c73fef6d7dcd706e77f9fb86c927",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c7df573ba051b71648863ebebf3d6858db3cd91935247b7df933356335382f83",
    "sha256" + debug_suffix: "3d253d43949c3853d76382ce5f06573dc8d21cbcd9d796b634328952136acdd2",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "204eddd1444aa2ed8582055351074ca42912a09b3e53cbf818a564255c8204db",
    "sha256" + debug_suffix: "40afc6d9f2d071a5b72f8804f43fa972fe21e64a390ba56cac441303b10ff4ee",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "fdd9a9229a2e51e0a944bbb4230e791e7daa854dcec835c4566907fe7681f3d3",
    "sha256" + debug_suffix: "bc0c95ce4bb3f66b4c08cbfd22f24a984be6ea8aabd1f3e5de24913390e8d717",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "5d00942b48b0684975436d8690fc88c17df76cc7b573c180066f4b10f7429823",
    "sha256" + debug_suffix: "e369ebd7f3291704e99990da5407133e0d9df30adfa42e4eb13469ca2b3bfec3",
  ],
  "kernels_optimized": [
    "sha256": "377ceb0664b29a463476787a25363210a19746b001c07f6388243be10ee65fb3",
    "sha256" + debug_suffix: "87325b6f8ce7413ef9d043545e3e5124659c9b501440467c494443b9585f079a",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "5841ab88f3ec9bc6729ec1f064b431cd1017011d95d083d031dbf47585f701aa",
    "sha256" + debug_suffix: "f5a93645f6d77fe6eefec509a5812204d4d43393534cfd60e7539563ec189318",
  ],
  "kernels_torchao": [
    "sha256": "eb4544027647c6740606b384a3d911ec5b645ecbaeda85d31c30d2f54656806d",
    "sha256" + debug_suffix: "d62b5852ec8222b1a1c1ee01dd9f958ba19d7565269409a2a7d9afea10e8b794",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "f59c22f68a1d777dd977067d4344255dbf25072f6bc05662b77405bed0426c5d",
    "sha256" + debug_suffix: "5b1b318456af7eaed7ba863e173c1f88f63c6ced7b9ba05b99c1af7695efe3f8",
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
