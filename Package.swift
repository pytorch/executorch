// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251031"
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
    "sha256": "c461cd0fa7b858372dd1e2fb570d1980e37d5eba832d091961111283c2053c1d",
    "sha256" + debug_suffix: "009063c227325cf16bbd63f948e84f3c8e2e1950a0ef84e7821b69b408e59e5a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9e2cf83954753566afcb9773fd464015911c36afd2c7fd3820edfe2635639171",
    "sha256" + debug_suffix: "216f6db31595a06bff652e99257f4a051eec3a19cbb0dea3295dd7bcfd558d36",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "da501189c71c07ac3fa45940210b957304a92303049288229235adeda82f7f01",
    "sha256" + debug_suffix: "058dae34df07e79354c1d0b3b65eee860054836610d4e39670b100c9c5eddf4b",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "2938ca86b7922d4510c19ab7fecedc8ed2cbc32923d4cf8642083fb1bdf756ff",
    "sha256" + debug_suffix: "873b556252aa5e1d3b91b0f5099df7821edc88acdabb9e77fd8b9a3c0c1bb447",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "bbdb32b7ab545f8d2115170aff008dc9eee45ee4fa271bb5fbf6ceb728011052",
    "sha256" + debug_suffix: "01fa57262bfafe2d7fe00cc95e4237f553a0d5df40abbe1a66a89ea9b11dcbba",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "96c23fec48c108312d94d5275df3614dd2253be7de6ba6aa2233a4ad98c4c8fc",
    "sha256" + debug_suffix: "48fbc975e8ed12a3cca516da156f4166be9d3e8850aa9535436fee7d53d8683d",
  ],
  "kernels_optimized": [
    "sha256": "e5d154ce5b7a994a82483cd47746fad40e322625283525abb08fd0910986ba25",
    "sha256" + debug_suffix: "41caebd400734123f50b15862ff4fd11fa78c66c32b730cab65110147bc3d351",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "4d8e9edecd52e8a36d82384706f83f921b36f195e6276205453ec4c8545e6095",
    "sha256" + debug_suffix: "8ccb9987381123686860399fc28a24ae39026b55b2dc8d0ac64a316ce1a690ee",
  ],
  "kernels_torchao": [
    "sha256": "ea87b9e48a930205404994b8f87b40e65ecbbe29e77a14f5b2d59d38c85f2e1e",
    "sha256" + debug_suffix: "09a52ef7bf589a42c9d8efe7a0e76149d0760cbbcff547b7c47837df9c45273b",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e0f5b37515fe1b7f9cdefa9d6470a03dfbe39fa9f5f9949af745871e37778107",
    "sha256" + debug_suffix: "76b147b7b7e1e885350a78b1bbb29cb812dba762cb010c08d68b86c845bdddba",
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
