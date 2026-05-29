// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.1"
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
    "sha256": "f92141884f1f9fb8036ceceedbc59fbe8259a5b41ce5767fff9fb17d07b88a1b",
    "sha256" + debug_suffix: "1be6168ec07a733d61ef06c1084036b4cfb4c81c244d77bf79fcf0dff3c49ff1",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "cdb431c85e342fc071f9d38aac1daaa5f94e6c0559a867c7a686b8f53f1f167d",
    "sha256" + debug_suffix: "e018d8a3949fa00831d063039b3c00b65cd22338b88af7ae7caeb96d1fd441d1",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "268f64a3159867ae3be6017e993aad047a613489a5604e237d8e8dde71522805",
    "sha256" + debug_suffix: "a87444954a59d10dc9c0f2c9d5741ec9cd62f9357133cb9ba1ac98f83a701227",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "ab38a5aecf1402a5963c9ef23b12fc7a08db32608a1dee4937c87222406a2ab4",
    "sha256" + debug_suffix: "ab2e9a9204094226fbb7f21ce1da75e3da95a04e5ec3f8bb2040e7cd3649ad0c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "0d6aa68058d90368bdab4836e3955383b8db5cff29f0c55901cc004a04a3fa4c",
    "sha256" + debug_suffix: "8c586eefe7fe2bef3ea7b2112ce7b9d9577d7173c17d3bfebb46269356555f7c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "5a4314f763a8f34c2a6c1e3befe77e78e20c37bf77744f7212abf3b2204058b7",
    "sha256" + debug_suffix: "0300cc85bd8cba3f1c53ab8a5e5ed2bc3f2319123b82978f2af4cd34b89a9f6d",
  ],
  "kernels_optimized": [
    "sha256": "d2b03ddbb6dc767abd7f6baeff3da5435ad3e60bda67a546ff1891cdb624c878",
    "sha256" + debug_suffix: "01dea087f34fe0198acb9dbf71bd4b123877a1a65cb6d16473044be89d6d0cb0",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a1084a8ec18f65b8127238d01033772e01744ac30fe7c01a76fc2b45dea3a6d9",
    "sha256" + debug_suffix: "43faf5841f71d8fae0de1e2e3bda08974a5148054afcf19723dfd9687e4d15c3",
  ],
  "kernels_torchao": [
    "sha256": "978608f6c5a427cb3279a97b07a0a72f02bd54910725ad57f0eccb3a4a753902",
    "sha256" + debug_suffix: "3fdffbe9954ebceef7344f8745de3672aa6b9f02eaacdd00dc1193a85aa18aaf",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "cc79da9eb28c023316622e109e0f07cf301677a189f4f59a95e626cb91ab5851",
    "sha256" + debug_suffix: "02c45cbf32cae576fc8290d06bb27abb27ade4f4e1e03cef64ad6b00305a9e02",
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
