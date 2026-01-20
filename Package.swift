// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260120"
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
    "sha256": "9d246e941d4d5aa0467b516309690cc881f2148bf3ef55e07edc7d0ac6fcdeae",
    "sha256" + debug_suffix: "027755e54ddf27034c3c23d026b69f2844244a0f87395b0397f828b3c9930479",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7d473f2736a982c524e442c4f987618dea08ba708eaa2dea8a2dba40756673bb",
    "sha256" + debug_suffix: "9c3d0cec3ca932c9a84b651e301f1b346c25a3667fbd9de0ce2e51e396450496",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "9cd71d728db2b7fc03f59c31f93785cc21a8bb2dc38e4a1ac8514fc8cfb3fc5b",
    "sha256" + debug_suffix: "e4564cfbc11551fe434e439e630f24662e8da90474829f549821cfab0865c2d7",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "b015d8aed09d87da8aa6b894e23481c020f622a10d2cc163a2a83c5d0ed30cdb",
    "sha256" + debug_suffix: "96551cc5b4bba33253216935473ad2096389e01230296091d9b318e3e2cd747c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "7d9e8cb1f12eb9d98333e5b71d1e859447b645669564ec5a3b578963c4d9bede",
    "sha256" + debug_suffix: "3b4b28b091b35b4baa5597f955d41b035c1ff2b56f86e9591bd9464e78d745ce",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "533590565019f7e05ea38ed7584a136f9c6302064244c540ea7cd3bc9eb36b32",
    "sha256" + debug_suffix: "c49a6ac5130f8c536fb2f18c99f8166b6157b0910931124d6b2034a9f1da5afc",
  ],
  "kernels_optimized": [
    "sha256": "2597808d32b74a0d4284c2a56f36d7fc6a0a217fd9011ea2cc547c7fed5872e0",
    "sha256" + debug_suffix: "0ae366535ea69067f231de51c9adc6658bb888c0738f5c262c9b322ca7bb5827",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "2f799c7713c7564383f31e1e51f45e8bf20aac80264de9b985d5ad160b2c3250",
    "sha256" + debug_suffix: "218a74c183bfdfd7573fcee7a0ad3e3817e13c622f1bce5014ca275d866cdf18",
  ],
  "kernels_torchao": [
    "sha256": "7aa47d26c2b2b43bbfb125dc2323d98cba1a94fd7b6530bb1bb99054150f4883",
    "sha256" + debug_suffix: "6acbc5e4c9de3c047716fc9d6e6a9e725039d6e059c81e4078814def45e6a0f3",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "ceff825c095884b23bfca199eb1ca31e188592651d5bb142ea90f3a425ee098f",
    "sha256" + debug_suffix: "e67b6ebe4644043fb2f02cc2dfe5efccd3d0168eddcaefc104f19cdbfacda421",
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
