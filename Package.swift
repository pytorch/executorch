// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260302"
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
    "sha256": "d39032753639a85c910e1984148f0ff4610fa127d16bb5526942e4091db2bfd5",
    "sha256" + debug_suffix: "254c956043084feba64c3372e76772c0f89f44bfe9801386a648ab065d48c4d3",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d964e2af8243d0afe33c262185f658184faadf745cd6f945646d5668481972bd",
    "sha256" + debug_suffix: "77aaf9479f8f66767eda8abcb04e5688eb710eb404072b98a1980c4d1f2d8f56",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0674a17020b614181b9f6406d55ff0616015a57bfd015d9acc70c7157ebb18fe",
    "sha256" + debug_suffix: "b1b5988415e50407bff40dbbe4829edf3be8e95f87477c0554ab65944877219a",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "b5591a59ede546f8d723bcaf1ec84c722fb67edfb1c3299d898c22bd4db3cea3",
    "sha256" + debug_suffix: "8db42858885a51951de09a323bb139f8710aa38a1b8545feb3119d00a42fe9a8",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "481ef9fdb1f7be78cbd4fe68cc6c88be17d7ad80fa94aa8f08ae3d34d0605e0e",
    "sha256" + debug_suffix: "437f5cc64f7efd9b68d9e49da1cfb5c8b82bf94090cf26084d558ac82c3076b9",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "2234d85892c94f1ff3e34b5e8292f2ed995e398e0d82c335561fcb6178e10c60",
    "sha256" + debug_suffix: "3efd4ab2911f9d8999785cabbd2c60f03d4ae6f82a69e6735aaeb979f11ff537",
  ],
  "kernels_optimized": [
    "sha256": "805432a45049b8a1e7b92f1a89fd82dee0c09452c4bbdbfb2e6eb7b29a0dd37f",
    "sha256" + debug_suffix: "e70259299f07f61abb3204be48645d3d797bd91fb70c0b7ca20c0ee661e9da46",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "314720864b1224180fc9581215402f835dc23417185c1c83871f9fda05464f5f",
    "sha256" + debug_suffix: "ec4ca297a87497385a223a8e7fa125344d748dcb276ed0602adcb96e9cf7f73a",
  ],
  "kernels_torchao": [
    "sha256": "b9b69aaa67ada9d122d06c65c7d3d7d2c0204190368e28bb3249009513f819b5",
    "sha256" + debug_suffix: "65aec928742084655b1de9c95cc9dd3429acd639e96ca5a929c8ce14297488fe",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "b518ab7b1d8ee517045b5490ee8c078c0fd23bad1994df34f065b9c5fe7b87c4",
    "sha256" + debug_suffix: "18e3e4c5b27d484a988d86361663eb36cdc66e36f9fbfb4ed70b767f8e656a72",
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
