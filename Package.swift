// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250609"
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
    "sha256": "c39e9873560548ade4139737a7cde38680c5d62af60e360f0b1ec2a03db8fdc7",
    "sha256" + debug_suffix: "ea5a4a70bb0d7c75c4f5bd7bfe319526edeb26f8fca100a186c08697a2878fb0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "95957c07873c6f2f129092128b6a8d303a4b38222f237911b6a5074a84cde4e3",
    "sha256" + debug_suffix: "c5b47682ef9ad88de6a24c6afcf248936ac0c4e18012656e8d6791410c290c8c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e796cf4d025856ab7bdfcbaf2dd7bf96fb69e613e1cc64e601f0e128c8d27c37",
    "sha256" + debug_suffix: "b81b087b7385881386f92a3295b88f2949cbe4ea46532f55d8f743bb118dc649",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e5be711c9108ad9462ab1a8a241fd280775cf645ac291186e088a3cc4ac96fc0",
    "sha256" + debug_suffix: "dedf2971a8b7229865ba4275017cbec4184743f8422fac0daa1b893c3f8c8be5",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "b08dbefb63bafb3f6e9afe067609250770dc4b70d68980d194f00b6b577be713",
    "sha256" + debug_suffix: "dd38f9faed863466a06d4c49aa66e94a21499fce104e25c3a12147796b51393e",
  ],
  "kernels_optimized": [
    "sha256": "6d5eafd492e98cb6ae793f0936fe4232493896d55b2d3278684c7477a7288617",
    "sha256" + debug_suffix: "560ba483e9afbb122c7aae9002abaa8583ac505b7040bf5d3247d4e404ac0f52",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "393a69d0ca0e42167390f1961185a5a34ad79ccc4f9abe75145fcbda94ed1621",
    "sha256" + debug_suffix: "3281cb3eb3ae34b9072a963689229d6495ac5e840099fb90ac86f935c2e4acc2",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "0b09acab5332e63c1849493edbadfd6298921b26ed5346655ac1fe05e5e0e38f",
    "sha256" + debug_suffix: "280a52645a83f5a7cb30c2ff7851f088010f152dcf205b13ee680484467d9f53",
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
