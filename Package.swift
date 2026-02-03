// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260203"
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
    "sha256": "743b38e20e7008d5865987c73d4ce83f025359c7db12960a2bb4da84be9fcaf4",
    "sha256" + debug_suffix: "a2654be491552b117117485d3c2d27eb37d487eb1ecddb8c758e62c1cec74195",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "34e9358bcdb5b4d5a8ef93412356ea2b5a9a777a95b936a2de3bcb1fe5c5dbb4",
    "sha256" + debug_suffix: "cfc00932c00eaf2934e6a77d7d77c4cbaad5db5091326a3ad059ebbaf3cfae63",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b72be33ac6a5f43d21bc7314aa06cc82c27ed99bb90b0d17c70ee078140e0d63",
    "sha256" + debug_suffix: "1862daf4ecfb702677b34a7afee475c4b4e89dac4197ed0ac3e36a06255e6cd0",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "ffb9d5959ab7c97a6d9854cda92064a1d6a6cfcdf8e30cc5b70b87257290be2d",
    "sha256" + debug_suffix: "d36167f1f4e158a2cd922936194faf0e1e5521e47761053f307f51c50ae8e079",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "39b10855111a15872a5170c596c6b1d19ba6f4ebd1f278505c759189958b0346",
    "sha256" + debug_suffix: "961b51b821737e5cd5c660cedb954be177590911b363e905b8211db08fa469a8",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "d43aba6b2968e23dc34b4c013e3b5dd4e9841d2686685563aae54216428d24fa",
    "sha256" + debug_suffix: "e39501ceecbede81419e40714230980522b15c7b94e4caa3503b890de5ac9435",
  ],
  "kernels_optimized": [
    "sha256": "0acc1b41cc167e66d4906dbd3d7e4e05becbbe3944ef3321c6eed09a0e8deb62",
    "sha256" + debug_suffix: "06ace3ea872f72296611177e54c9b4dd5eef8aeed406d3191a648bc9da0cd8ed",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "3b3087d9d370a8552b9181333cd23f1856c6b6e3586bf1defc60119c974a9db2",
    "sha256" + debug_suffix: "8c91075e1045e8e7f0afc6eae3454669dfe5c7099c2c9247f7951a1fc487d9df",
  ],
  "kernels_torchao": [
    "sha256": "edf2569cc3888504972f356086aa5227d2f3e3b44c3a371645d57fa27bf6d1de",
    "sha256" + debug_suffix: "43cf165403152babdd9ef9a378973ea93eece914702f3e802e81cf1733d51dd2",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "53afdf4ef4e72d5f10163294d32f92f29a23626e9365d0206d3a2821a0f97c6c",
    "sha256" + debug_suffix: "6ff0b0c58a845fb366229f4076faae1ad63efcb271d49c7835b74677427823c6",
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
