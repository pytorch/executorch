// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251008"
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
    "sha256": "329de83863bef16574b17dc5ba375d22854098f709a8b26c099d66d14d06b360",
    "sha256" + debug_suffix: "cfa95c7ec43259b6edaabe9a23173dee6efffa083528ed19a3addabe67d0b849",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "47d8c309eabe39642b721f3ac66ea60dc295c312b0ffb5dfd5383df31f17cf08",
    "sha256" + debug_suffix: "db8eefb5d5c33c32d0dc2ad89c0209593a16ee4cb7099eab5d14ed0cc679b40d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1ec4c1e65b53072af09894f16f2b6bf296244bd177ef3793ddafa181aa0e542f",
    "sha256" + debug_suffix: "c169b9d5365b37b9136a44634165d8f26d05f7c689f66842591e8db85b118e6b",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "3f374470cb188b2fe44f216e9f25a5a158c4a0ea1f5558139d0b22d75c3838ad",
    "sha256" + debug_suffix: "dc7a38135691e3c8a6afb8df946b0bbaa2fcff80c57b8773fdd65d09d05b1780",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "2410ec90f8da8b607130527aef59130fe34dfac581fe175520dde2798dbc24ab",
    "sha256" + debug_suffix: "62cc5ab8888246ef4a177a4eecb0f2d123e02c08c2bf6e066567258b475e76de",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "9306a9dded0e02a1bb0c572ca1edbdb7d6c526dc226c6bc5b047e89d0aebf359",
    "sha256" + debug_suffix: "8e7547dab4fb03bc4b059c0ce682ccf7c110836507cc0d24b070a38c9ef4dcd1",
  ],
  "kernels_optimized": [
    "sha256": "1b7d7b2e809b57993f037de91e274e8ca640061e8d616fc65f28c324e4b28dd0",
    "sha256" + debug_suffix: "82f813c4453223d9e5bb413a839995af607dc2d74f0d364ad2c11e6df55bd13f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b15cb8521b9e5f8b56d911ffee8718b232a4795901d9d3ef2e35b9bee323bfca",
    "sha256" + debug_suffix: "99a5e9f8f8cd8d3b0e8b92e86f933fa804ef6ae7cc59f66a8e7b0df15e39ce75",
  ],
  "kernels_torchao": [
    "sha256": "df59e1878ed576cc5be696b16b1560d803413c1ea7f73e825da3f7cdc787d930",
    "sha256" + debug_suffix: "9234e348069dd3f9edadd681241ee0c1d90e6d56bcdf03e81107d2388562554c",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "5e68750d04c7e3ed04c9caaeb8a5609ff28944284f6a2633d67d9c1b48e94d5f",
    "sha256" + debug_suffix: "9db8e85b54be2e3a501686663aed03b6c1d3a38970601ec0426e15b369f8b5ee",
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
