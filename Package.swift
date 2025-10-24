// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251024"
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
    "sha256": "54c55a52f6b4f0ca467ccec7448a343c85dc4b28dab5f6f3b599a0263b5bfa76",
    "sha256" + debug_suffix: "7a5a5083b74e498887fcad5e627cd6bcfb718a912042be8aa8bb0885f8586d2d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ac6a278c9c9ed389dc14379c2cc6bc49d271c46036a0bf65eddba6ed60f9f12c",
    "sha256" + debug_suffix: "d87e76bd776987d710196635352e62a6ea8ceb15c32d04ff87fd60cbfef6d853",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2e59172098cbda7d69774375cae6e141ee70b8a4fb9451062203f74dfba24e1f",
    "sha256" + debug_suffix: "eb8ae69e3f9698b3863d53f6953f431ec97da291722960961b41dba0b6d940aa",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "aee163332084e17923a4ac149e2f3e7db45f2f557909b8b7bbe2b7ecfe984741",
    "sha256" + debug_suffix: "cfa1112927cebe375809445a8a423d8c34f4d3c6f840a3bbe4cb9e8d0819f9bf",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a1c04f41b52b0c3282b209529e3eff4f89aef9cd02eac8ed5ec6fc0dedf433fe",
    "sha256" + debug_suffix: "6976c3dcd1d6a6de543de1be7594968d9fc3d16b23c30ceb8a118af9a014e691",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "d7a67074a32c0dcc8bcda20774a618e722194035facec2eda8d44afb2b86d3a1",
    "sha256" + debug_suffix: "07798f2bdc532df95b907f2cafe45a42eaf7c96f051403d320bf50f33eca05a8",
  ],
  "kernels_optimized": [
    "sha256": "173c95fbd73e2dc3f0d9dd158616870924b2b0f1cbb588b1798a17983e95be0d",
    "sha256" + debug_suffix: "a3a4f96379ad0fa47fe5f0c694724330e983508a5fba38c804f9068ac1537807",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "34e2bf86b8e6cb8c0461fd7f9c7f78edbd8d7786f5c74132dc907389df60de1d",
    "sha256" + debug_suffix: "6249ab001df09752fbdbbf44f40eff8ae1bae9b6f8da55be0fa9efa10b6d05d7",
  ],
  "kernels_torchao": [
    "sha256": "8cd40cbc9309afc103ea2675b715af01bab9ae96f15ce3c0be454c7a6a0e4f6e",
    "sha256" + debug_suffix: "bbe4fff4bdc5c16fe690e0ce4b4447307e9db7b81fee81b4f2ba7bb0288281ef",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "7f8d361eaff2f420171ca4a46066b3e17a460cc06258c384d0c3f3f5950adae1",
    "sha256" + debug_suffix: "bca8c0e9562b6a9cf07ef8658ada8dc93546635953d113e11a66861bc4870359",
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
