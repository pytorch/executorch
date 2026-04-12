// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260412"
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
    "sha256": "4ce457a17870dd2b8c41a0e75e47ddf6cc47f5f70f0220080cee6d47a50fbf7f",
    "sha256" + debug_suffix: "acc182890ef29a6db9041a61306d8ed3799492e58a73313dfc1d4323908a3c8a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3c1d697b1c11e0441c387f427b48231d49f9dc7b6e9d3c16b9d8fb631bf82c7e",
    "sha256" + debug_suffix: "ef5a62e1093afb89414959d7d638826ea3b871de3006a5859060200bf339dc64",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5ed758b8b80fb9113bd6d2579cbbde023f613fcb540a58765650f93ca3646cd5",
    "sha256" + debug_suffix: "f15ea419427717b55963790241387baf3de3d7c421b40f64987454c865ce50aa",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "56d27d5564d6a31c090bd989f3fb040619a3d9209183639b3a5be6e2b70e8dc2",
    "sha256" + debug_suffix: "10c1c9105ae08a23ce1838d60b5fda758c788dee80b5ac7f945b4528e1cb0caf",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "f883115a62ec22fa3c7e875c6b26c270fe171324105b8b6edd20192a5f2803f8",
    "sha256" + debug_suffix: "72396e8ff3a1d402305b3ebff83d8e86508984a81fd80de681b14a170ce0048c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "65a7e246fbb86ed3890e7a1f655ad1da76c4ab016dce303d544ca3af0e1daa6e",
    "sha256" + debug_suffix: "896383f131ff0d1adfea3cdd4e3b9a8a03d2c2d805165f1f5a6285a7dccd83b4",
  ],
  "kernels_optimized": [
    "sha256": "50e31d7adb5ff943728093c1fcd91acd26a3031a3c9997cb1567cb0fa1b71e09",
    "sha256" + debug_suffix: "0d34bc3477d812b4b2cbbc084ad11180ba15db3c327998d14db0486ceab124d2",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "6071fe750770c38b7ff3085431f8dd5ad9558d968f5779ac3da0393135d88207",
    "sha256" + debug_suffix: "409e185157ab01c9eeaea3be4203b5584f5c42bcb2c132c7faecf98dd8410335",
  ],
  "kernels_torchao": [
    "sha256": "c1a30b9ba53f886ec55b7a5be7a5ed8a65cf8c541e80432d5e4ce342785221ea",
    "sha256" + debug_suffix: "6c6a34199965a218134efeb86d4e8586ad6ba443660b0a1cd54cfa00cb9edc82",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "ad8b26cf183f8867700cd56777c63dd1eef28a1a9a76e03da2286b4d7c491be5",
    "sha256" + debug_suffix: "e9b77b92f75aef3bda13909603ed978579d076d697397264449b3523b69613c2",
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
