// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250927"
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
    "sha256": "09c082e7c9b79a0cb6c6627800a95ce3e4359e79b441d80ef5137266b701741e",
    "sha256" + debug_suffix: "9daf9e7373da1729e4b10cebf2c95aa17b321a31d546d3c3f7543d2b2a21b2a8",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b6e32515dd440b064abd6fe8c4354866efe19ab98cc5876d24dceb3857ec0a23",
    "sha256" + debug_suffix: "0c484abd371def1169503da3e62cc860b54d683b5571415f25e64cd13946c62c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7bcdcd764272aec202ff0b82cba20c052c833e034e0fd3458d4cd6219d4bf51c",
    "sha256" + debug_suffix: "b04e11403de25b9b78d33dba5e754a2ac37aed21f4796727fa6310f72b7b594f",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e2ac864dd5e1a3545f67a679931239a6cda0686abc0c6264caeb98d6e7a1bf6d",
    "sha256" + debug_suffix: "c5dec21f0bc5559c60ca8446bd8c4da8ce3c3a80bbfb8498efa9bd03fc8aae54",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "0d7a070073dbcb17ab092ce7829848f03699dd5747aa57f99d05b5f7c5639021",
    "sha256" + debug_suffix: "589f744c9be8844a44b5c0070b5647bb703cbf94242e95dc0354a2a42f7a62e1",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "bfcf4968f169d11267a679c168eabebadf3c50ef33d9ae4f8e027ddfcf1fb427",
    "sha256" + debug_suffix: "3a230bfe4422d21e745b74e0f91e25b8ec64db07df342883f4a1e431c2e3c1a4",
  ],
  "kernels_optimized": [
    "sha256": "821f5dc974a9ba54d5edfb818909d1d88dc971a3250cae9310283692c8135c34",
    "sha256" + debug_suffix: "7fc6ff229601db6b38540a6b68c06cd005e6372894111d2e4bc18c37af32ca69",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "cba96c2abf4b6091257357795b71363df9b669bb895479b9a33585508e3620bb",
    "sha256" + debug_suffix: "25f6f4263878292f79a136505ee1fbb4f1e2ecce31f97e1f87be410704aee05b",
  ],
  "kernels_torchao": [
    "sha256": "81892b62e64ec9332f547059d0d4bb141f9fb1e97f2ed97f52cd33dd107335fc",
    "sha256" + debug_suffix: "edeb1e4ced48d2e24c42d285a37bb920ddb013801c234c1d1432031eccd4e942",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "1edc5e23af81a742c9c1815fc623e4ffb82a25d23ab3c6e1403c4a61dd8e64c9",
    "sha256" + debug_suffix: "4356c4a1c1d655590c95fe79230affe329b3295867700170725a779bee3880cc",
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
