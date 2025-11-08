// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251108"
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
    "sha256": "0cabec293b4dd224da06e57005e605446f1c72c7cdd51a9d07f916432ab00acd",
    "sha256" + debug_suffix: "caca2a3a528b299b86127185a79a51488b901964a3b0dff9f0d6ff77d3cf5219",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9949d8c5cffea37c5ba5f7348a4784bdfcca03c7e00a6704394ae31c5801b0de",
    "sha256" + debug_suffix: "1fcb002828be516c0b68e895d0d10cde0d5285ebbf0aa19f54bdc9d63ebccf89",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "440daaac84cfde13ad60acf78d02a3ee9f1952d8df745fe8f5c05011c81000bc",
    "sha256" + debug_suffix: "b1b4282def610ec09dc0901662ce7868a3da5a90b01ef89f731a751e0d9d6a44",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d88ac3198b0d20b3523b3c59e413c9b8058682e4d191ce0201d29cdb9ed4c94b",
    "sha256" + debug_suffix: "8ea856766fab9da49337d7dc73ecfd985db64ba17ceedf8215aa01b2d5cff420",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "d037ae36504c5affbf29532bf451acd29b9136b4e0a55b5a95123e1676b06609",
    "sha256" + debug_suffix: "ef6670dbd0f37c138548a2a9303463405740b2fc6063edccce9fd5da3f934a75",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e5aa2a2018c442ac27104cf731cf1a586507bb394e2496f6ab26893f047602ef",
    "sha256" + debug_suffix: "1aabd9f4db4e2aca5c024445393d9d6b9a82607224f1519e2966931c1b17ac62",
  ],
  "kernels_optimized": [
    "sha256": "e972f8c9a6878986d01ffe9a7a8372990a1eb526a8ecc7bd4bf50e1d3a885c5f",
    "sha256" + debug_suffix: "a753ba60a0c4b24e344c386b00e9b8306ef567ba19f5caa0769de0c9211b93cc",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f5bc01fe7835c3de32c049654f16130c67ad551c500781b7107b06b893b47802",
    "sha256" + debug_suffix: "ba5aa55b6d6d6e37ddedf2591e696495b85fb725f935202f2894026ed9585a6b",
  ],
  "kernels_torchao": [
    "sha256": "52170bddd4b354ffe5864e945e5a322b437b373a9b3ae97d1e6319b432cd8591",
    "sha256" + debug_suffix: "a73a926a778c1979ef703e62188fb0ee4205fef1b35dbf2b5704c9b4c0f54622",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "44bc21407a76048c47d897bc55be67cbaa0610310109f426e80b15af30935ee8",
    "sha256" + debug_suffix: "b94cd8c6e32d135e29450e90590352ac437a4c4b9c6325ac4a621b0aba066b05",
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
