// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250925"
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
    "sha256": "af69d47fdc0a4d1ec7ed25603561a0808b0bed95f134681c6f4b1738698afc1f",
    "sha256" + debug_suffix: "9a5a36a23525368bcf0d4a40e6223ac37bf117ff44d81d4fd7c965ee12b34993",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7c21aac1ca5d2be73680b9c326970aa91e2e7fc3a2ec411251001baf2d43f028",
    "sha256" + debug_suffix: "31b9b27cd01d13c2eae93071c2af839996ca61e499497ae12e8d114fd5663eba",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "951ee0bd24adcfd15b140fbaadc0088b4eb651ac3b2e750ef5f721fccee14135",
    "sha256" + debug_suffix: "b432e9d9e25113b3b449f63bfc96ad80621a6cd972ee00518f72a622a2220e52",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "1ec62548b86837590785773a2aac4926f965d0c5b8731067a2e429f18ed7fbea",
    "sha256" + debug_suffix: "97e25facf8cdb4e3263f89adae2cf66569b8a31fad16d2519c1f581d4ba8dcbb",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "0b9ffdc3434143d6d4e53277b4eb1905810b946ce12c44717595fc77e6a860a3",
    "sha256" + debug_suffix: "743150f919ff0242af0de12839eab2f3bf70620a7f43655241049599dc9699bd",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8d2bad22a845bbbca508ee6800b9001019bfa4aeaa044edc132eae5098a66ad9",
    "sha256" + debug_suffix: "633f64b906f7bf22500e1838e348db1524484e991b6b531b99b6141f3e851dc0",
  ],
  "kernels_optimized": [
    "sha256": "6446e4f082fa458d219230cf5dc3d04af1b73147c706b57aa69c8f3b835095b7",
    "sha256" + debug_suffix: "2aa2906f732476c1dc0b8ef1d1b63530575c20eee86ec27e86bce90f88274e40",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ef7a3af47ed0d516615694272fd12360e1cb4be52eeca3d4d1c70c7617e5e843",
    "sha256" + debug_suffix: "e60993b061f39274a7a7ba1c91a2582c610efbde72de29f2ff08d79db03a77bf",
  ],
  "kernels_torchao": [
    "sha256": "b6c6bb9e063d314a5514aa8a36398592133390c6b3653c563eea9e8538837df9",
    "sha256" + debug_suffix: "297dcdb8e8970c94cd6dc58c8236f97b39d1b758dda06a6168c62172625e5f97",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "0de8c7e9357bf87bb6512c5168971ce408f256471127b2eb8431a39e790f902f",
    "sha256" + debug_suffix: "a2c2f47d4ba40e4c6191af34af74124ca19743af09b74a2564ea44c94ff458e3",
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
