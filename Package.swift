// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251230"
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
    "sha256": "7d4446b47e4eab84d608121b2b67f85d4a9de0d48f84e4a12ea92503d28b807e",
    "sha256" + debug_suffix: "f5691075879fb92e831bfda84047774e4dfa3effd7a2d0dd453282f100f13543",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4ae3e50ce08520c5621783c897ad72287c17124ee39dc335c811a6bc15c928ea",
    "sha256" + debug_suffix: "77a78a27b0bf6ebf8172bfa8f3a1be7ea7a5c02e0e0207f65be92e9ea3592cb0",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "77baacb52812fc52961f0b57f4b2f23447f93d0b0d7105b72454f53cae907637",
    "sha256" + debug_suffix: "4e1b9a05c71b0b6080502aaad22769c705de033b7813ae48fb1104a2b35b39ee",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e46cabc5e17aadf6129bd74f8e49686653f86e5d3a61addc5e1296dbcad99d1b",
    "sha256" + debug_suffix: "d1393b93fe31fdfd31ee13ceff2f1cf47ec9129e992c2407bbe93f5612c1cbde",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "9733dfde8cc4ad51119fa0b90dcfdcd9179558d440e0e689b082f2209490414d",
    "sha256" + debug_suffix: "525e3e5cd4e8fc731ed8dcf32ab4f673b8d52ba39f13f1ac027ef4c7b0218ffa",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "28d08c890baa03f5a76941b5f49b86e20ef4bcbe8ee9d1c753e815bb83e33289",
    "sha256" + debug_suffix: "889945335d3f8a8f0e248a24319d527f014b3f421f5715a19bd5d557d54aa12f",
  ],
  "kernels_optimized": [
    "sha256": "26a6a635e2a8ddddea2dac06028967bab5f0024161383d6f456915f8d180c7e3",
    "sha256" + debug_suffix: "348fb7c2bcb6abbe6a83d9db270a375dd98e6c75327cc306c98cc8d80f9febbe",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "7927388f7a927b66f68236d9b42de487806db32c9b6264674296aab8cd41f1ce",
    "sha256" + debug_suffix: "b6070d2d9ee2bebabd52a53d67a6773a86d529d0e425c0a59e1daf1287cc3750",
  ],
  "kernels_torchao": [
    "sha256": "95f15d7b96fa342e10dc800ef3388402af9ec3f4b38a70d4911c097cfe693e2d",
    "sha256" + debug_suffix: "813e7c6eed982569a5341be55bad9c3f2f563eafb0ebf5867526299be6cf1e37",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "d9314364fadec61c77680bf2e4efa0fa4505e3e5188ae0a6b3e865df4a147f43",
    "sha256" + debug_suffix: "1fa53fce8d67563eb80383ca95d775d6d23b58922c2940dd4363cf884ae7fa09",
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
