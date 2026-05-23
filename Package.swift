// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260523"
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
    "sha256": "2369c9d7f9d240b5af4d68242d9b0015c8e33826096d54008ea89f05c5991caf",
    "sha256" + debug_suffix: "425645d75d5450329078cbb71f0ef793b3edb17e11f3e57d19450bd2210eb209",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2ec80b7c86e6bc6ea0ac703638998d7182788aaf9f17709e3bd3739c7ccca987",
    "sha256" + debug_suffix: "f164414371bcdf61d7048d1b678d06638440cd218daee5370b9f84ec924b98b3",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d5c1752113999d9b4786cf08db6d35e2ebe621106393ff1c4ff0d98f6a6ed77c",
    "sha256" + debug_suffix: "0909c606f846b45dc3e99935bf7229c18be6277912a064545f89d0c7e445b25c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "b6d3176ed0cee649a8c8ba9f027dad050be35d0d50b352fc2bb2f6043f6e878a",
    "sha256" + debug_suffix: "85ab25098de1104ac33618fc04d6e00ac069eb5f4112e9e853eebf45e77e9140",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "905e45de7f8c6ebdfa27da959c463e0fda48b98a324f41354d5a1b7cfee0f307",
    "sha256" + debug_suffix: "df45dd28c61ac3cc967bfc877e6189036ba854b950a927961a87700cf21caa06",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "846117d1c943d5735f69a1d29e65a6fe84651a0fabda834b58ae97fb83424e78",
    "sha256" + debug_suffix: "0db8f94b7cd366beb969b59a35278224fde64a79f6bbd14b9548d564a29c0052",
  ],
  "kernels_optimized": [
    "sha256": "8ed3cb66545d6af91d4926bee4ae4a4406bc7bdb5fd4010a2c7c4302de8af354",
    "sha256" + debug_suffix: "d35f6cc9631501bbe3ee433c25f43fa35d046c7ef76b9737ce7cace09c0927b6",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "76093274d322ab1d3d36aaaec1471fb04c9691d7a40ef305152eb74dcfe0dc61",
    "sha256" + debug_suffix: "0c8c9e0acc3fd17208a5978563761773a7d0ddf564cd08e115acfead78858f17",
  ],
  "kernels_torchao": [
    "sha256": "fe5798c2eb16d7641f3a684e512014b73c1e54e8b846bf852ff8b6cd6379f42d",
    "sha256" + debug_suffix: "cad1372d19665f734e9ed0594502cf05054189191f4dde38ba6632094a14e2c3",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "b261a8a72be69d702eed79aaa70b540e6af7682b3a33a918cb6b580116938007",
    "sha256" + debug_suffix: "dfb2f9fefacbc9ebc2f961b65b67d7a59e18465b5662e973980b11637625be91",
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
