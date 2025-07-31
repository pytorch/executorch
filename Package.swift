// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250731"
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
    "sha256": "0a85da3a764e42d04a6c2d7bf8e214e043b853c1a1bc868c0595f9a74b06ef5f",
    "sha256" + debug_suffix: "54fb3a85131b31426421a37b5c45f25f7c06d35f2f6216d8b405828421726069",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0dd1cbbced29f2721a8877787bd1776ccdbe32f43d40e8135844b289a7b09bed",
    "sha256" + debug_suffix: "0294b6fff81baf66108bfca72a56fbae5e7583a9f258738e48fe679213759828",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a658da69fab29d1162853893dc6c31135ecfd2fbff62638ad0ff1f65d4ce47f1",
    "sha256" + debug_suffix: "3c5e98abb29a55f551db3237abe796e661c0326e6e74b84409519d461f021150",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "da02726b112f3cc81531e8f741b6c586bdf91d1deb88066190392ba39a4914ca",
    "sha256" + debug_suffix: "3d3dc9790d134e6d0c72d04a430632682fd5e2d38a2a6c7fdd79350abadb675e",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "3ecbc3dee363279e78cae3871ce155c6354e6b7ba506143a186f7ec1de81ef93",
    "sha256" + debug_suffix: "10d4487acbbbff2c0d5d4edf973167353c83d672dcc181f2844b89132a311600",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e1f4e9f978f3e1de9aa3ad5895b9ca5474178348c426119b236eb8f83cf738c8",
    "sha256" + debug_suffix: "bdb85f169b0dec556387a70ca07ac772443a6d2dcf6f1ef2eea0c070c173946f",
  ],
  "kernels_optimized": [
    "sha256": "c2b5b8dba4bbdd63b31e1b4db2aea73d5d7b3420bbc8c17fbdbf759d3b361313",
    "sha256" + debug_suffix: "3c8443e1a4bf8e264857ec5e34e1395daba0f43378a6f8d3cb712bad2e94e899",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "5b2b1bf4d3bb7e44edeff9d44bc70d6edb23f8d8f03657cd5212549b0bc9c08f",
    "sha256" + debug_suffix: "1ac24d7d30cd92a5d99698d9590b3d75d5bf2fa0126e9313965c212158537dd3",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "6901bf952685cb73ead9fbec8bdfd955adbf6b2e33c3c0cdaf0dc277095bee73",
    "sha256" + debug_suffix: "03ac11f4e09d71bd392e2754200c018b7adbca064feef0fdf4849dcaacba5e27",
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
