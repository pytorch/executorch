// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0"
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
    "sha256": "af3b39a5359c63036e174b3c141f9a01279a1270890a88a1de2f27d35288b485",
    "sha256" + debug_suffix: "2258a4ab04f8a3df2316322de977c494b2f01728ef05d38bad0c9056e2ba7015",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6cdd7799c5856def0d6e657bc55536ef7feb96eba3cb4422ef93291f87fa5525",
    "sha256" + debug_suffix: "c8334b92486d0ae86f7a50f6eb826b20de59e2337042f57fc1afe8f3e1826671",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e0a229aa5726b78b95bac2d414080bd3b439ea106885084f510ecbc172d6f808",
    "sha256" + debug_suffix: "21ec394286d5f374bc876b698a41a8bb18a0e3c043c4002f0a3140db10f0a5df",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d7100c10243584f560ab1701d57492786ae5a46e6b47529dff355a4077eec295",
    "sha256" + debug_suffix: "cc7a9eba1511fbcc21330b21cec15d9aeb72fd686e88765f3533e0b2b4fb7d2e",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "5bf41ef4f474bab0e7c34c32da542131dc514b4d7e556b679e3f1f3a3a5513ea",
    "sha256" + debug_suffix: "30aa81df0c4374f8c703a0f6ecbeefdc088f8f26738332b28328bbde0d06454c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f17a3e1dc7c2a5ff8930b1c67a9ebfec60a1ead22663d1dc115663b545a8971f",
    "sha256" + debug_suffix: "2304d46a2e96c3e8aae0d923c20d61b641e802e0dfb5ff87b435321b24e512ac",
  ],
  "kernels_optimized": [
    "sha256": "56771a8f34783984bf31fd72ab58b45532f4fa7662d33a458a9e3caf816d735c",
    "sha256" + debug_suffix: "eea0f8b00d85724aa9c74baf9b7a38ab8b4eca44c7496f3cf296c6002edd3be4",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a22da412f2d765d55eb05719f6a80045b05d0adb8bdb3e6d8123f8827ad3123a",
    "sha256" + debug_suffix: "061f246b9e0ec71939203b07c0382be98ddc1fed7425f642b0195e764595d43f",
  ],
  "kernels_torchao": [
    "sha256": "7c018925096ea3aef1bb72654c8ce152918edb3dac6c37e071f2e4d99700b7ad",
    "sha256" + debug_suffix: "9e057e83e50fc5f73cd80b4b8166c436765110bed3dd8591edd8afea70cf04e4",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "d045f305d8d8ec6763415f1afb5acea7844f8d8b0fb6de8351b45978543569d4",
    "sha256" + debug_suffix: "1399a3cbb2d889aac129dc8887e624667decf33a1438b591bf56ee209d78a529",
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
