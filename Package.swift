// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260703"
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
    "sha256": "f2c2188a38f082589267c71eb28da9a86ee4990b522ea57464b28dfa45f41ae3",
    "sha256" + debug_suffix: "40db3faa1f6fb04dac3d61fd6a0e80fd54dbb1885b113b4169e84db854f028c2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "89cad608960d754f406b5d54ce1ce00c1405966e8bf58dfbead27f919bbc44d2",
    "sha256" + debug_suffix: "9e94d0fa3364d4c501ed723c8bf9bb6af82067060b7d30e363e5def46df0109a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a4d1d3308629a4f420f13bc261bc27cc8c2d6171b77677c2775267ae8dab1738",
    "sha256" + debug_suffix: "4e9af749a4b1fc4a4b073df490043b7a19f7dc7007a236387c6aa594aac1226e",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0e2cc10e62e8d2e054faec88a7f049dc990890a330ba00fc153ba4b6126fe427",
    "sha256" + debug_suffix: "af73a6b6bc51bb5c5dbc9a61d0423644f75488278b0001a246851336916bfa74",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "5c8073e0d7c9116b8eb4891be79772dfdb1a64661a16dad0f68d0af3f7e7d33e",
    "sha256" + debug_suffix: "01c126750f9823decb99089c9ecc14514f9a4fad846a4f0a8af9dde881ce3e67",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e9aad752634fde1b8ab516fbee0c181e44f94fc56a36700e37ac66a5e86646c9",
    "sha256" + debug_suffix: "8ada3b55c065b8e8e5974bf392bf71828229b221ab9513a67e55954eb5c9d988",
  ],
  "kernels_optimized": [
    "sha256": "325916f1430948da4a222539263b2014b86dfa7fe641fa8f3f18ca0a660ceab5",
    "sha256" + debug_suffix: "109b1ddcc23eef6b93b13162a78e663c412b007b22655597eaf459037fa5118e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "173260bd66a2caf9e27227ca1665582717fc5dc936be31cbb483515892c53f04",
    "sha256" + debug_suffix: "b35ebba62bda62f05afc65b1c4d82fd2a7575af4baf4bd44148dbe5d3f1e09df",
  ],
  "kernels_torchao": [
    "sha256": "92f34b6f535b38dee39a38186fd928b0d36f79b76ed40f06eb7377914dd72c57",
    "sha256" + debug_suffix: "a9181b1337c9ea5d82dfabd3aeea49b6213b59d3bbb6218d25f8c58f3082bc95",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "09b65efb53f3babd620e5e5ea9be78b0a868097619d0d7f0bbb706a1b8a961c0",
    "sha256" + debug_suffix: "5b385ea8400d850aa7a72f48420b0f407813ab22c560a647ea3ad1fa44b300ec",
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
