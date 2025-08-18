// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250818"
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
    "sha256": "ec15241efced52d57d8a35d8066235c99af31c7b96e475de2167d00d2ecdcd80",
    "sha256" + debug_suffix: "16572b64a4c6533f00bdbfded7499edd5d3b428e197464cb606c328a3d799c1c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8a8f467340371f9aabdab1af7d6e92f98d11630ec0ea4d0a000a581e41f2242d",
    "sha256" + debug_suffix: "5fe38af6870b70e64120780b250a04efe9e76f33b5e8f25b24cc3c9c21a8d703",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f667357180577860ef65f7f9376a00bfaaf99ee9e1df87bc92b838c6615d235d",
    "sha256" + debug_suffix: "1dde12561affce307d6c27ef6def99a5ce819fc3701a743c01cd2bba2ef944c3",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f0a0ab097f22e5fb1c1afbfdebe761a7e604f27b1045e4ad89e8446728822cd9",
    "sha256" + debug_suffix: "c7cd40a7255c962b48406af71407e206d4b573a6b9dbdf0d3f80c63ff0e7b441",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "818dfc93e329c0e932fce7b640a75e175daa11512fcb5441b8979ed75ae5542b",
    "sha256" + debug_suffix: "97e05a0d2bf01544b3894204a21e76931016ff98c24997ef62867da2f073c6a3",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "9b01cf36dc58d72cb734c586625cb72d1250dd6ac606783b5a3ef0b04862960d",
    "sha256" + debug_suffix: "60856325421be54c08b64c2477d61fa4ece8074469e4d7b4019de237c3290e58",
  ],
  "kernels_optimized": [
    "sha256": "1c960aa57f6e521c84a5172af2b07a462c2955cd0deae924af9fcf7753ecfd0e",
    "sha256" + debug_suffix: "73e79e630a359013b0944d3de7e07b60d8dbf672dab96dd7bd573f45f9da7294",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d1711a0d24906a8b0d788ea6b204bc07171b3b2a40f64a217cd30bb48cfd4472",
    "sha256" + debug_suffix: "0cde3946d0294f6ded313922eaa69307bb4bb03d8a5e8415588452ee84b60379",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "073d4418a54eaf7bdc525bca1fc0d4e011a8dd51a733de5610040a8b6ef2d71e",
    "sha256" + debug_suffix: "aef4bedfb71e9f42a5db9b6d48ee5356b3f97678901c22d29509b2a89be6a636",
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
