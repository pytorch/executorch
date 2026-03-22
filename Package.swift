// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260322"
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
    "sha256": "c5daf875b67908e9176ca194d27741c2f8667acce28e4e3a61afa09b6d64b2d7",
    "sha256" + debug_suffix: "0cc040613230aacbe5908eb1eeb4ddfe66054454dd21fae7a6200347bfffac19",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2678a814eab8e7e32953a0e08b88e51d39ab0a735f967933d7d7c6ba8559bf03",
    "sha256" + debug_suffix: "c6989eaf2370f3151d31bf7d78a678404275089c18a73fc60832618a1e7f679f",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6d94dcf4f464884bbafbb36d2de4da4503e9c22b6462c47b5de077b10be9d894",
    "sha256" + debug_suffix: "7d1eabccef73c2c76d622ad0d89843e836a5c425af3ec8657899ca9500ae2975",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "00d554ab0a5d58bf0aab1ad6f172016b5aae12a7f6012e253ad590719f554103",
    "sha256" + debug_suffix: "a9b7fc8bfb02097d99d589baeb2648d2bc8b766cbe6338bb57f251cc319eae1a",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "c28251ccc0548902066daaa237f8b101c5808e3d8c3691fecf9ccfe51fa6e936",
    "sha256" + debug_suffix: "6890a89683d709423c48af4a15f9c7cab652b079543988ecd77cabb357f29d76",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "1b0637d2ed3214476b871f34e5d8429f3469dca81a9ab13180bab2eaa2beab6a",
    "sha256" + debug_suffix: "2e6bd20f2521df2ac2b1a20f60080ea70b20da2d5645bcbd20bc36c3bbcbb10a",
  ],
  "kernels_optimized": [
    "sha256": "2943adefd90a5cfa3958c4c45e489f9a9c6fe0e1592c682058571cfc579fa76f",
    "sha256" + debug_suffix: "18a49074dfabf1763021450a84864ab3b7eccff18f42cebace9d7f7583831707",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "2a4b9cf1b8aa1519401f2f0b9f5ee2be756be2f264fb3d9c6219869c8047a93b",
    "sha256" + debug_suffix: "d4a904b89b0db4c3daff6dfe6f8fe224735d53bb209ff57f03f2bed3a1692f00",
  ],
  "kernels_torchao": [
    "sha256": "6c140b0ccf76b97b427c862ca624b42f050cc0c183e6ab5ee0245bbae12c7a4f",
    "sha256" + debug_suffix: "b816f4da0991fa416472ef2dbdc4d75daa174b2871b660e6ba77c0f280376b08",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "dffcb0e2e3f86210323aa37fae42a3a7724def868ee840a22d78b82ad391c262",
    "sha256" + debug_suffix: "c9fc11dc2bfc4d3cd0323714a84b89ba3ad28ee8585f8f4efdd453945a624dd3",
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
