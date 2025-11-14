// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251114"
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
    "sha256": "4a69581fb1b37f93a7c32ce5ea87afa7e9cf9fece30863f9d0911358c6ac36b0",
    "sha256" + debug_suffix: "94a5709cceee3f7963d24a76e06fe2ee4070f6670e66c3aa1960c2679a361f34",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "acb2d7d7162ab936124463927d7d01eef0a54f7b3e803313c7b39c1c197d8810",
    "sha256" + debug_suffix: "4282a29faf3a3ded0012dc8273c548fa7d1919bb5d327792f4e94cc5d4c34ca6",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4a00677e71f88ee757b4149eee3b61f373ebbca0b35c03fd55f6ee4658632fef",
    "sha256" + debug_suffix: "9b466ba044b4ec7ca1cebd4b586a17203c8d3412152155c11d24e06107b8dcf9",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "dda8c27fdf3497b8ee21f4ba492b81bb180bf0f70cfb7440b07490cd7c6804ce",
    "sha256" + debug_suffix: "f75eee1a41385d2d474e9af32595742dd0f12e8cc086ac1cd98963bdffe3d491",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e90167b79bb14593d702cdf17f2d4a9fb37845f59669d790886db2989c958632",
    "sha256" + debug_suffix: "13c5e8d82b136a0155b3d316b013c68a790f499f8ac19ac2b8d45f0ee38a156f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "207542d36161c8263042d2fcf1b4165e349fcb5800bf8cac02e1013157254f92",
    "sha256" + debug_suffix: "acf9f382f4b60291b3b289fff3226aff52cf380f8bd9168cca8e117e76f4ce97",
  ],
  "kernels_optimized": [
    "sha256": "6793ee5940cfe7414b9c9a9244d42b02d2122095d222375b95fc0dcb52fbde2c",
    "sha256" + debug_suffix: "ddb0e6525d2185eb0b9d38de96ae9df7eb5777b2abd22a46dd3bc98ac724bcdd",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "59ad62e0a9578307cfe9b3440d8e1dda86fc731381eeed96459075f90c0bf11b",
    "sha256" + debug_suffix: "26a42f41661d3ef109b7716c8c6805fce253ee065f935cd987044734b7b90044",
  ],
  "kernels_torchao": [
    "sha256": "b9d47838032038027641d7ab8d167d288d2d1fe3ed74dfd7f779df26c2f6f9da",
    "sha256" + debug_suffix: "ac8e5451cd95e94ea195d13166772d3e3ada03092fa4931dca9ff2a7e958d2d5",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "9ab17d711fb1ad4bd1b5be81a77c19e199957723abca449f16973653db65664f",
    "sha256" + debug_suffix: "b286d0849a42c1c022734d0a1c8c9b1ddb130d14881aff735abeb18bf5fbde26",
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
