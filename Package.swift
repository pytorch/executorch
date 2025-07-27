// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250727"
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
    "sha256": "4d047014ad217849bf30717c5c05fc889b38823aee772f2a1dc5054cfde63272",
    "sha256" + debug_suffix: "3cc39a223138df27367298a43f5cc8f23e1c4360f02e8aeaa2aad2699f435ae9",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "062570dddce194b6ca7bdbbacd6da0121c81adede2661d275eeaaa4b5366d655",
    "sha256" + debug_suffix: "b79ac35a9c19d6a746ad89f6aad4b9ed04d41704d3ef83f657a9fb2d2d70e270",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "bca96dfe83eb599ac84184923f7b00663fb370068195306b8818ddc782b5f6aa",
    "sha256" + debug_suffix: "9f57b73482a3ff293defa8bf0e29fdf86d20ca769428946b284ba5206490a313",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5540d8bfb5a5d6957881c0e399e63dccdce98fc1f425c7e3f26fe665510f762f",
    "sha256" + debug_suffix: "45358c63f7935cb46f7956e1cfa1299c454145dab490853be61434106d6b2a55",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "6bb556da3182c91c573ca6156f9845f51757c919ecbfe0843496e3b24d50c9d1",
    "sha256" + debug_suffix: "5ab601eb78e12794b8d7a23b2f90174109251457f2cd87ec443a5c60e65d6833",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "756b0fd0c733d9e5455e3a155917e9f01fa3082b322736e0c49466e6217bf170",
    "sha256" + debug_suffix: "0fec211675bdc8d4d8d4f2e6445be09e3670989a740ad55c1b372d697d8463bb",
  ],
  "kernels_optimized": [
    "sha256": "0101b3ea696d84261ec1317c21ad53653ee61482720e137aa207e97af43ee337",
    "sha256" + debug_suffix: "556070f6eae0861b32707cff5b2917c984f55ca84b77967deb12f4488a142a54",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "2e2d820c151a09ba64452a48e716a7bb589081ecde974991509275f88bd85d1f",
    "sha256" + debug_suffix: "bcf1b9adee27d4add410f264114cf8cd5caa3dcfd012cc4b78c22da45e646483",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "6db1e15d7ecddc1f6f6b3d81b5d2d8b5d8db7a42a60c4eec2a5397f337436075",
    "sha256" + debug_suffix: "2f52234c326a49ae745b01040bbfc6a292222840a79a89142f9f3ec6e18d9218",
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
