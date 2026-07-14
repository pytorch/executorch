// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260714"
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
    "sha256": "cd63eee491a03ee0eea3dbb10c0e2ac87b3787357d5e31638ad7c2322ccab638",
    "sha256" + debug_suffix: "d698b9051663aced803e1fde6fce8b0b7ed2adbdaf50df62a3532de6180bfe2c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "251e00627698a24e13c6d5e4b700128755e17a46aa054806e00cb429d063e96b",
    "sha256" + debug_suffix: "dd315cfffd315639dd892b6a639ced7b01ebf858ca72faef6b262196ee5e58cc",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "00b65d47e6d37e8a57a47f321ca90b2b4ba4db8b380883a31ba20ec847a344fc",
    "sha256" + debug_suffix: "27707612df749bab5aa52c12a1bf4bcfa47a2d5600d9e5aa121e66f2c68a5649",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0cdbe8ff55dc2e33cc37b7c4852849965fef33e1147ee40b911ab21e69be76c7",
    "sha256" + debug_suffix: "4c1a849107d8d58c35d42afea994c2297f1d1cffbfb0b87bc996c6ec0ee065e4",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "7fa3f3365e86919f72cd730d4d7df8b995540699698487f588e99c85544c5b92",
    "sha256" + debug_suffix: "2e3c93f3e0fe4abcc005258a0622d364df81da0d362f3d4176ec14414b70d47b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "2ddb8621da87b190b340508b613399c561488d1857e294586d5ea681da49dd2e",
    "sha256" + debug_suffix: "60ae61916297415f060c59a09200aa61032c9d832d3fa3124f4711819e5f1b84",
  ],
  "kernels_optimized": [
    "sha256": "375c41d9803877a7c73e9043696b659b493a4bbeb531cfa9212f2294974c84d8",
    "sha256" + debug_suffix: "a20f2b4f43d4364643c1481dea0e1794fe8da85b8c4bb2952c09a96720f3b711",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b227cefd1bb620da65d39a8023927fd4e9ef072db8561866a7ba8f1469638c07",
    "sha256" + debug_suffix: "3695a704da00eff4861eeed9e6f743ac9516947a16b3a6cd29a9a2ad82f8b968",
  ],
  "kernels_torchao": [
    "sha256": "6f0aaf094e98b64a5c1a4bd346d2b068db87399b5d6a0aaa65ca3f82fcefcd96",
    "sha256" + debug_suffix: "7932d02a2c3ad179bf7d9ee922fa4049ccff66a4682c1ea8a04ac40bc84b2c72",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "b975666c1dfe0b674f4f337d8136e7664622911aa0dd86ae617ba9035caa2183",
    "sha256" + debug_suffix: "f2a8eb3c0153cba39919488d553f8a467985df67c4d6eafca124b324796f3422",
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
