// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251215"
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
    "sha256": "89c2b2d6eba40aecd4844624791887ce6a467b65a8fb6177d47ff9c9619e79b5",
    "sha256" + debug_suffix: "fada43ce9e11d52d7c98132f7adaaa0dee5aeae7b1490ab5753ae77fa86e66c4",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2a26f84c22a11afabcacaf8c9b48d31527ac2822002734efa389f16aa97855bb",
    "sha256" + debug_suffix: "94b9680fe47b4108118f70841f86df3e28c3e5cd233b6498f210996b60e67210",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "cce46507ce95ec4e3b6515343c548217ef064b9aae1305d970f380c7333a667f",
    "sha256" + debug_suffix: "52229304e6b3a4f5e3c4990426bd55e0ea470e58156f9fe18083599b8dadb484",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "10f90b2dfc52483682e9cc3937d3b1a723fe068d27adb04791dd9f7906b88d79",
    "sha256" + debug_suffix: "68e36d9e88cbd20d4a6044a25a1f28a5dd8d541f59068a7dbcc784cb14b2ed44",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "49c4ba1d0ead466946edcc0cf784a90b589ab107a753dc5e3005caa5943c2340",
    "sha256" + debug_suffix: "a88564bcff52627863129075d1f2a60207bb7b50a210dbe91c0e611d9ecb6064",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "7d73c5c963cce44e695bdf3dd17864450770b650039b4e97518c4f64db9b42c7",
    "sha256" + debug_suffix: "5af33a9f8aef6d4f19804e80723763c87fcdcf0772fb3dac61a949f8f767658f",
  ],
  "kernels_optimized": [
    "sha256": "44e6077f99a73852e6360e80756fa03ac90154d581d21fbaf4ac111272c0bb42",
    "sha256" + debug_suffix: "4c075f4389dee4f2d5946d76a962ca938917eae3742155e4de2a4009d57a0774",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "47f9915bd6851028f2d41b90611557c95cbc8e4b02b83302508a316ba2af5adb",
    "sha256" + debug_suffix: "ea3cbf2e699e3af10dfeccd9789f63228434fc896d2c3604fcc34f0a4e526807",
  ],
  "kernels_torchao": [
    "sha256": "40185164ca1e08cc26a3581fdc2cc5ae6d730ed34016fc10c7e611af3a491c63",
    "sha256" + debug_suffix: "6079205f229d67580ccb9cf748da1cbcd43f606d7bb92377988c0c790569954e",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "94a51ab960e546f092793aa26be71dcf30f74a836cb6b11c04079da37bb5c7c6",
    "sha256" + debug_suffix: "f58a840c00569a40c4bb4df018fa769691c54b2e847548166018b5731b73c779",
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
