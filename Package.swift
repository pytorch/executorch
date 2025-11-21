// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251121"
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
    "sha256": "99bf894ffb112968c6fa8347ff528d5b388de809e84733b97dc149075e7be7ff",
    "sha256" + debug_suffix: "986547f8a9cc493685906b727a7800f74e2ad971371b49032f9bda8999aaf978",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "321018dfa418c213c925959b04c1d08aa6a530096d974980cd57d14bbc6167f1",
    "sha256" + debug_suffix: "a652c0abb6e5cd1e40bb8ee230106535c7688074f9234010aebc9892b676f4d4",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e08d8a98f90229b6de93ef67f11346cbf20d625a3641b3a529a3170307b60b2f",
    "sha256" + debug_suffix: "57802860dc79e270d1e8e3cff46359ca604df80221ce3ac4df34e03dc1e0f0e4",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "7796120904b24bbed13db712fc8cf74b8127ce9ffee5f23cb5a2f06b8cec94d9",
    "sha256" + debug_suffix: "7d7501c85bd0b88cc11e4a0b0df4d461108bb8bb6efd842426ec5a5dba755c82",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a59ab5748d7b71e5d16a0f66e6c9a5fed29cd6bc0f81c041bbb04e240b8de3b3",
    "sha256" + debug_suffix: "edb0850c2cb6e5e28203d9b8e716d6c48b3045cd859cdb50b76c104d3da73f84",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "60dd14b88cea4a9a89dd8d24a487ff7ff447c2bf31c78231ac9eb785a242c6fd",
    "sha256" + debug_suffix: "af5e6becfa779205804690177b7d20cfc9730e0bef04e21b04ac1b49f913075a",
  ],
  "kernels_optimized": [
    "sha256": "423946179d3319cc1572a2416ad5d75bdd53c28661afd02f825c49e26d563acf",
    "sha256" + debug_suffix: "8f2ecae1f1989491df3e69322673cf1346dcbf78f4619afb0044e9e157dbac0b",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "cc9aa992f24a937c46af3fca5c3435cbb3557677f0a4b0f0d4ef1ecce0ec50dd",
    "sha256" + debug_suffix: "36bd9b3b6c08b3949dada9dcab5761255a6bdd19300b732cf18cc1003d638f6a",
  ],
  "kernels_torchao": [
    "sha256": "9f28066c254eaa601562abc1c5aef7f7187e327c795d175678f1b04216356778",
    "sha256" + debug_suffix: "329f9831e29047aa90695355a1d97a97d0449b7ec4a3dd612479978beb16c0bf",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a1d2623a5520e01068fc839fe75dda97969210b9e544834be80f5e0836c3a89c",
    "sha256" + debug_suffix: "6f0c5cf2cab4499f6e8416c38fe3d3800834158b00f1af2127b766a54a6df01e",
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
