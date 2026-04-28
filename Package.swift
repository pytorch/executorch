// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260428"
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
    "sha256": "f100dfefc234bd896cab8e7368aa8b113fe5c5dcfeec32fe4a122ca7be27bb1e",
    "sha256" + debug_suffix: "f4939a7065dcd1fc680aede3c4dc5f67618029677715c350e3621dc1d478631c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "236486cff02a32d6ec315adcab17beb6e10c15fa79da5b2233b27059821de500",
    "sha256" + debug_suffix: "a990d8c4dbe5212197907e0c50023b38f2d53f7d07758dc0472fc78beaff39a7",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7d1033b28a811b68a302e1198cf4fcc75a14e5ae4d41c5fd8d11db1365db311d",
    "sha256" + debug_suffix: "71d817c83dd4892296416b744ea91c5e01e7e9637ed25fa0bec5ef80d25f3de8",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "46eeac1a95d82688faac7eb8dcf90596176b76faea9a83b25b48b8f225c6c599",
    "sha256" + debug_suffix: "46a3983176f623499b272b2fa171d0f7c5aa6ddf4aec404e2f77dceafa9ecd35",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e650f98f8f89b798236a4b8895b09b131e3117c0081c76e58ad3d83597ac732e",
    "sha256" + debug_suffix: "c742292d9ea2b2b84917b2e09ae1ad4edece6f8d3aab32b8860dd62a14cda9fe",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "cf5ae288fd8c84ef8375c1a59a4c8658e69905e72f18b0716c17a66bd863167f",
    "sha256" + debug_suffix: "814c89cc31dc1fadb7bcd6329a5935109b83def2d7e5061a356138268d685192",
  ],
  "kernels_optimized": [
    "sha256": "e78fb5baa8367fd4107abd1d561d1c65d49083afaed887682be514097ef47f11",
    "sha256" + debug_suffix: "418e8005200f3c17ced1f8ba83460d117b1628f22a3746992e02a4989259b656",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "1411564927d380215d947139d3f2a525caf36d1f87e6a861c2fda56a2054d7ae",
    "sha256" + debug_suffix: "850c2cd948798bae66a361509ab3308c7d9a34870e36fd82358996310d86983a",
  ],
  "kernels_torchao": [
    "sha256": "5560c788b6d2d5822742c89c20716d4b626bd833b8955bba46334f4eb48bd091",
    "sha256" + debug_suffix: "ad3bf63404f2981cb86dd859d718e73d25cfe2f083a91ddc713866ce6934592a",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "8243d071ab3081f0d2d995f75fc3cfc35e089e586a6c4ca1c93c2faf4d8426fa",
    "sha256" + debug_suffix: "0216f1d78370d7102af73b99e577fbdddfb73f74ac2a39bafaa15e05b91a363f",
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
