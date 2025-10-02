// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251002"
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
    "sha256": "f4afc04029a841d8d93b5bffde3e3453d9db3a26708b1c98e1d6fbc4985aabf9",
    "sha256" + debug_suffix: "3e268c7852f3109d95bf1bfd71dcd0e4aa8f071f4788d1064c204db2c64fd119",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a85da83ffcaaa771e5a8be53992e692d72f88781f068671e1646cca305a34550",
    "sha256" + debug_suffix: "fd15519e10950a66cc5bd91fa6af097f4547b7de4eff69eef9f5725437e97f61",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2056f4ec8a5bdb6b3aa1bfbc7558213abd4eabf5267d90851ef321293c4237cd",
    "sha256" + debug_suffix: "a58f83f46edd990e4df25c1765395fb3b022bef26ac9ae3e39ebaad0013d79a0",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "44003c45800f25f5ce3423f75b67bd6c3c2f16a188909b31cf0f02dbb799ccdc",
    "sha256" + debug_suffix: "aff43273e9496dbef560587975aa9b4cd470790cea9557087839eabba1c3cef1",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "28de0dd15378b46fa1c11858e9c71b485a74031e5c9a0090ea5022932d136e84",
    "sha256" + debug_suffix: "4ab34f2634b792eb52f80ad6cd3124d82f39176c758fac3b1da6cfde836cf0a8",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "06f918817e3a1753c2c90d3efd9a02917869971316ce5d8454b08d01ae203c76",
    "sha256" + debug_suffix: "e36bf9b8c1a16280860b1a37d89fad7ced086a2df9cae358bd02943b403b9557",
  ],
  "kernels_optimized": [
    "sha256": "4c5300a7f617308089f9e841f9436f19d94852f3a8640baa17d8a331edd697ff",
    "sha256" + debug_suffix: "e5f0e8cd565868675dca525ebe03ccb43ab01256aeb262b10d136c1747c36652",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "1b94aa8527bec40da02a48c063c3e34dc7f4ea5d4acbbac4d7188bd9981b2282",
    "sha256" + debug_suffix: "999e5b17c2e28f805bd90b69aa54883cc7d9aad97d11dbdd4d4d17238ee15aa0",
  ],
  "kernels_torchao": [
    "sha256": "ad92b62335955e3bf35379b2c1bd0d2fdbc061edb056aa10abb4e9b0cfb7f859",
    "sha256" + debug_suffix: "3e32bf093649bfac66659c07e3cbb1fd99344285c1ee069515ab67c2ba255be6",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "1bea895c22c9780b2c1d0b86d67465c714324fe8c63487882f4806e6d7058fc3",
    "sha256" + debug_suffix: "15ed2806aed54bd7e5fae53e15b0f904044687996489bd780bc623d4754b671c",
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
