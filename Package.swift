// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251228"
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
    "sha256": "e7b47c8f2f82b52866af96ef0f8e45e28769d3de7efc182de7e7f85111afaaaf",
    "sha256" + debug_suffix: "290086da44c15a5fd8965bc21a585daa6c2092979571206aed8eadd7095992bd",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5315205e516975e0a5c966d1d6271b94ea63124d8075df272c3a744559596e00",
    "sha256" + debug_suffix: "8d5992b7a8b0c721bdf7e7a23e30b2fbe76954b29a624805dd7a7d6b08dd4319",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "9e420268741a784a851e92ee7155ec555fccff4fd424f0e5d0de1681003ceee3",
    "sha256" + debug_suffix: "de78b7fbd952c2279fbd5ab7d2996e20e89bcc9d97a97e9c7c156dd88d08e785",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "ad345ea57b38e0f953bdc245050e610c023bcce2137dc804226f2ac981a47eab",
    "sha256" + debug_suffix: "0cc5536f2838e8911c3135af2a50b3285dcfdcb3d5bed995c2442c9063b63054",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a4fda7ee0d1497e0e3e4fccb6652c68dfba46e9a7cf2b86b635f2d0a86544a67",
    "sha256" + debug_suffix: "bae0e0ab249f42d9e7eb90bc7304639d34d9dc2d41eedea55c2f77ea22de86bb",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8f13e473449b7067123f3f38ad38e2aba2d0b6296117257553603c63bef64c66",
    "sha256" + debug_suffix: "f45903108d0b7d430a8b9f3f25280e6ae620b6e0abca650212767b3f392313ed",
  ],
  "kernels_optimized": [
    "sha256": "bab79d4796cdbb35c6d39879729e2847cb9a4620b241f3f65614f68bca3003e8",
    "sha256" + debug_suffix: "e871499d0d9de4bbae401005aef7f0dcb2aa6c57ef5b0f33791eaee21cf04033",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "05fa07f3c232195a743a0d9c5b7b19457c22b8c6ca601a5aae814570ecac627b",
    "sha256" + debug_suffix: "d589dc81afc4b2bc767fec49083c1fc0a0d9fcdfb439845291e88732424403ef",
  ],
  "kernels_torchao": [
    "sha256": "73653eeeecbfff6774b9b819cffa9b276c3ad1f25e8f3876efb507acea367419",
    "sha256" + debug_suffix: "871cafc897249b05133b2a606ce7609a8218fc4a1c71fd2e8f09e9d3252c66a3",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "56ed30b5113f5ab40c9f566caa894c65e210dc5bb7806ffd3fda5a12cb9e36f0",
    "sha256" + debug_suffix: "0b89e787a22488828f9cb561383e30206d00a85f678d8b3ba0cc5c537b2a8cb6",
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
