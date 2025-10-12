// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251012"
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
    "sha256": "c79248caf4430353a175948c5aca94a5076d8147c598021b8177e7d6b8dd0078",
    "sha256" + debug_suffix: "f56e837cd8c01ecfcb0cccd53962ae708bd19980a88e25bde41466bc8375ca3a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "491fc41cec49cc5830818776916823add02fd180ce0d4b7dac3df730d775484f",
    "sha256" + debug_suffix: "aa476cbaaa64fbecbbd2a20a3880bc6f5cc39d9f32f79ab293c3fd76da9d8043",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4ddf312c259251dc67c9df45ce7b3632513e1ce4ff6244a06efb5cb92c040540",
    "sha256" + debug_suffix: "84e03fa1fe0f7c22644e5eb457442333e469d61c241461f390060060156efd5e",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f2e353a40178979fc9ef246a3817065d403cea6f5e6164573f5871a7cbc4cebb",
    "sha256" + debug_suffix: "df75557725c4b0f06f6c99dc8e6a205bfadb7785ac650bd1f7020390ffcf7ea3",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "257a3f5e2df33aa02928a8e5bbbfb12e565f78e846dc1ca7706428c1ca103ebd",
    "sha256" + debug_suffix: "2cac5d342e807dc6432e4761e5fc9b687207b0bfd2e838dae87ff157eeb7e9db",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c72436bba274cbe1538ca58796c37990dd234d6ca7e32f6a526a25ad9a262841",
    "sha256" + debug_suffix: "7dec860b999bc2b93321e5aaa55d1e4f0945723c7cb48e5915035b3a96afe97b",
  ],
  "kernels_optimized": [
    "sha256": "f3ad4121e50ee2930027fa4ae4f183b99f7013b0625f6a9e78493ce081307b9e",
    "sha256" + debug_suffix: "c615d79cb174116c53b95a91979ba068f15773c8bd933edad4e737a8f0830191",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9295ff445e4da38f6e971f5eb0751fe41c508765d3a7f3b7464446c058714ad2",
    "sha256" + debug_suffix: "9345f611b1e0c0313f2169957f466558f825e51e8fae8a15a2064c9d8a801388",
  ],
  "kernels_torchao": [
    "sha256": "45ad9e0c4cf532c25e380e48865714137a684ea9a32c3734eeab741f637f1237",
    "sha256" + debug_suffix: "92e71dd6dce0bdbcebfbe1608da51e17ca1d09da65ae678c2ab010fe0a23e846",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "ca067857a0028888b5ae6872a64b50d32e19e6167be5bcf8d2411af4b11ec761",
    "sha256" + debug_suffix: "e6efff450315fd96407eab67c1a6821fac89b8947c1857b7fa4fb4e627543f8f",
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
