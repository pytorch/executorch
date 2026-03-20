// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260320"
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
    "sha256": "35d556196cb51880f0af6ff974e927934d9fe9dcb1e39dc06603b7c909983c84",
    "sha256" + debug_suffix: "35eda0ecf8f6f08880dcf663a4000754598b41bc793032db594312ce5dbeabb7",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5b13172ed0a885982f5223350cf9d925235a2a0d188235f4cd156a786662d196",
    "sha256" + debug_suffix: "45840f6f53e48e2aed0142a7ac31bffdee200e7c67c7a1ef9464ebc0d615cc1f",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "aacc1257c476bf207d1f305c8587f49a50d0d3458ac2fce6d36f36b5317fb600",
    "sha256" + debug_suffix: "f0c4b5bea15ce30cd5d808f8dbe042412493aee35807d4eb1cebfdbf5b0a64b4",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "750bd5263c6d3f1ac9652c8cbac002e184094f1a90edba9258672cbc153349f7",
    "sha256" + debug_suffix: "fc1373d93b91f7714710b75dbb05fedeceb71d0f31128c4d822d3e53b8310380",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "eb8f374e395b6d8ad6691dcd619097d157131a02fd86608e7aa1a52aa8410c65",
    "sha256" + debug_suffix: "978c06e05ba35caaca86d1a34b5f2efc40d3058be9a8df6d779ff21f78b6709c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "5d211b5ce7bd35282a3ab2850b00996265c6640d5562c91d419600a0fca49ac6",
    "sha256" + debug_suffix: "adb32d16c6b79cbf7b239c4d10a86db11450d24d5f35bb43e1df5600c9fdc62c",
  ],
  "kernels_optimized": [
    "sha256": "d692499c5735000a50fb4e30839fbc2223f0999fa5f2f73eb6178a0e174aca54",
    "sha256" + debug_suffix: "0198e76b20a399b9d8d9476ac7d6f3cb68fb886c10b424be62802241ce2bf60e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "75e7e99f3dcd02760f362c3abb3dfc336d604f93d4da16cf92c615ed44147ed7",
    "sha256" + debug_suffix: "d0c11870f38cbc697428e0cf91f007f3ca6226621a2b0a5e2cfd21b1733aea3b",
  ],
  "kernels_torchao": [
    "sha256": "7d00e7b9d14bdcc88e48568ccf8c39e4e91dbf4eba434e991372f918960414cc",
    "sha256" + debug_suffix: "1eb527eb1a6b3a5b88a5ed37d7c20a86dd8d741c1415b74475fbe11c3fcb199d",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e2f121baa291e9833490b8f29f35614b5dea42c09770643f24998e0dea2d6d76",
    "sha256" + debug_suffix: "32b5bddc51b14628c0282f144fc716433189f1a2038a0614250691456f329d72",
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
