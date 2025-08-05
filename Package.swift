// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250805"
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
    "sha256": "6a0da2f6741e5d72b73b7b37085520fb2739e81b3f28e39c32f31c41147cb463",
    "sha256" + debug_suffix: "1a14f6128493b6033dd586adbb6f3b50ed994a8f3d5deb068f7b99bfc1de3ceb",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0c0ff20545aded1ccb8f37b9e364730602e86779835f83e7d4e175521e035187",
    "sha256" + debug_suffix: "2fbc8743a8c23222e4152a14bd309f2117539cd81e96728a08fcfd7eea93e060",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8e42386d87635ff3d02e6a209f30b6787c3b7202a04703a8a0ea10be449cb1a6",
    "sha256" + debug_suffix: "3ac0d077a6c219f152350f70412da057632d78a1ddff80d5c87e4c77f98f19b1",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "a76de8255bd3bc20027ddce6732c06d1bc60f12a23dd108947fcd02d52dc563d",
    "sha256" + debug_suffix: "611814d30357487f1fc7e3335acbfcda7bbcd1b5f04d86406c70fc6593c39954",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "ef635df37de091bbea11e71cd3f0ddc5635a07f5113848953d9141eb57686c9b",
    "sha256" + debug_suffix: "b16b96742492732e434b55b1d5e2631197db6c7d660df84b2cb507890dde3f67",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "48076e765d284f589778ca38a9e66d13539a2ede33b1013cf58c1054a0e5e093",
    "sha256" + debug_suffix: "d02af2e6479968fe7dd557de74dafc15772f66a0463d98bec024bbfb7db14c4d",
  ],
  "kernels_optimized": [
    "sha256": "0159c8868949fa96409b35baf6d2858ab71b5e6f75207370e61caaeab65cb3b7",
    "sha256" + debug_suffix: "0b5b80fd7e6e131fd974c4277bf68cbdeb5ca1b0eaf254a1461370b73ecd432f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c58b6df515f6d201012ef66c56c673f0bb9a1599c608410886758888c52e8c7d",
    "sha256" + debug_suffix: "2e20f3e6b5c7e43422a1f8d113443a1b8ca641c3e77a61ab88d47141d90b5504",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "2d1e3c6202f238a57aac8542a3c8ff476ff629b40e35c11186dcf1a04acb77c4",
    "sha256" + debug_suffix: "c39c21182c811de4d0765a9ce6842bf79ee09fc4da39e30aaa8cf1de4b487609",
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
