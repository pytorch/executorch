// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260222"
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
    "sha256": "76dd7bc08fde109a663884bfe6a2c57ba7b9741e364fc657deffe628302cd6c3",
    "sha256" + debug_suffix: "98a5386747265d871a51d182996738434f45928d05fca607e0f2e5a8bc3d8d2d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5117ac08f952e795e6ecf9b2682822e03e1ff85342f60842fff6747ab930a0bf",
    "sha256" + debug_suffix: "f066fbbecfe2b3f1e8f009535c2559036685ceb1a33af683dd8bf6e1db8fecc7",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6cde0104c67af3edd31e51f485248b69e3b02e4ba691cdb5f790d9cbb32771ff",
    "sha256" + debug_suffix: "f70b24f08cf8a17ea4c1e88bab9bdf5e13bd3bf71642890f3074e5c08a875275",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "788cf8d04a43d37768ba1648edee1a836a31f1dfbeb4aa896c74dcd9ff8cafb4",
    "sha256" + debug_suffix: "75ce4c8c31fd1b21f26b02d7e8f9ed63320cd044151e63fc788b5a9225ecf646",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a259f7c9130377223a0555a5b3666c26c4732d2cb5baaa5f720a0ecaeb0df85f",
    "sha256" + debug_suffix: "b8c8321ca37947b111edc389f7a830161e9df3a227e9764c72f8e286672a0de8",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "5f16955f1070745abe8eba16274abfb73bb3f474c18a3de1173236c3a36cf9b4",
    "sha256" + debug_suffix: "6c1a29643698a913cc9657d4b81ba904e77a4fc073c3bd2fa1b97f68d41a54dd",
  ],
  "kernels_optimized": [
    "sha256": "7c6947e9153839707cef0315aea1bfea11010231e457edd15e2be5dbf0f5fb9f",
    "sha256" + debug_suffix: "e4b68c0df55f5c4ccad3158cd6b3e2301cc9ecf0d977c8709100c5bbd54e1d94",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "2c1508f06a644ccaf721fdb4627f0ac9ba0c7d4197e5bf39b8bc601ff1d5b08b",
    "sha256" + debug_suffix: "067e26a443eeeadd7dbb81dc12e28ff07c2dacacabb8641e4566496f453abbd4",
  ],
  "kernels_torchao": [
    "sha256": "968dd1b4f33d75b2eaffa604d820fe030a7faff8d241e2602404b6a9d8c262e6",
    "sha256" + debug_suffix: "cea110fde366ceae62ccd00d8350296ee5425e0d2e0c40cac6ea8325808d4286",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "5146a4e94464574fa22d664732660e20ed63de00a2a4603e0e666725e81ab123",
    "sha256" + debug_suffix: "e0bd14a1eb285f3cb937950a4ba6ea46853af6a918c2b7531b484d55e396cefa",
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
