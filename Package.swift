// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250725"
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
    "sha256": "7b53e1416b0d3d50d2e0d66cfd91c5fb325e8e9a8bed71cdb6310d55f1849792",
    "sha256" + debug_suffix: "651a30e6ec026abb0931a1f43c1bf09823f42a2a059c05467d6bf5f4a123ac71",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a25b1207020dce06f5bf82489a8ddf138f2c91610be133938640c7d57dfcd1b3",
    "sha256" + debug_suffix: "9d5e22c0a4769047fdcece0a19a5d51adddb6c83d428be9330b4e1a7d5a45166",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "dd4f39a897961fd1feaba855f8e5f369fa48dd552ada2d8f20342368a1b557c8",
    "sha256" + debug_suffix: "019d44e882fb93966ef4d92c3167e80101822757fcf008a1a418a410cf9a000b",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "fabe9c487599d21556bf2459efe97c8fb217db45181431d4fe621c85450451ae",
    "sha256" + debug_suffix: "8f2852b03d2226dcff0722c36de484cd1b598b24283a2f81c8b67cfe570df912",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "9fe913e71f62846c0e0ce756db7bd225b4ca7ed64db12c84072f02f558db4caa",
    "sha256" + debug_suffix: "9140a0f71f73f4e17aae20a495b9a023088e06ceef83332a237d7513aa77ed5e",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "467b7623d3ec01c948c72efe99d208a2612073e9a727836e47adaa8386e14993",
    "sha256" + debug_suffix: "cb9a8993a293cd98be74b22ac57e4b4a51bec40764a637fea8527168149f3726",
  ],
  "kernels_optimized": [
    "sha256": "c3da36c82e94913a9e11b709455a419d1b85ab774cf7c81abcad818eb135b56e",
    "sha256" + debug_suffix: "26de99f02fb23fd5fed88adcc3b14d6d10597e0b825de7269080190dbb069b79",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "6b8b869402065038df3ccc48628ef3bbb890659cd7e4b718053724e3576783cb",
    "sha256" + debug_suffix: "5a976bcb9848a8ffe631755f2be1481408bd1c299bec844477d6291160142efe",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "8d287b789913b23804fdde68c116c0398a43f77d0ed8ce6493af740eea18d0d8",
    "sha256" + debug_suffix: "7b3442e853302f811e958be8b20389c1f394e4df09b107a169637daa01a97a8e",
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
