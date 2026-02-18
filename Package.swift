// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260218"
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
    "sha256": "5f3d8a5783b7a4c2a2e272008bccb55094bbfb02f8fc6774fdb673b959682fdb",
    "sha256" + debug_suffix: "374a606d16945fb01ac137d3fb0b106835b095ad0ce7dae7ebb1b225aef2d0f8",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6363269be00b0a14d642b8ba8f2082b9239eb016902cf306df99c187fc1a26a5",
    "sha256" + debug_suffix: "afb96e76678e3f82a6a9c6778dbf53f07ea7e965d3eead4a6f3600db3e2598eb",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "31fc28f25a80ac94e016cc7ba19e84f43aa05d1012f0d94bfbe6bc8c9191788c",
    "sha256" + debug_suffix: "095a2afa7418f87ea741d8a27a595ff1a132a50ee25764af221764e303dcefdf",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "eb71087ab6a07b06e6201e6a5bafb672046710aeec428d6fb2347412dd9851ee",
    "sha256" + debug_suffix: "f5109e43db9e3851a6fd0fe39049231c69132f32dbfe208f34265f018c592ae0",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a4c188ffe0206bffab4f0b0dd12e9cc7c47490ab4b71d77865803a5baf999456",
    "sha256" + debug_suffix: "4c8a91eaa25f11ee2b93e087fa16b47ac556af58a2a63d6354765225ddb7093b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8f641633d0fd76f8648863791fb5f25faa5029fa81415f42d53411f2d468df35",
    "sha256" + debug_suffix: "2c6200473b684f8be2eb10ccefe5382f994696594b9541bf813d7d775afc24d1",
  ],
  "kernels_optimized": [
    "sha256": "853134e75d86ac84d154fedf2206274fd51fff811b914657dd440a8e6a6f24e1",
    "sha256" + debug_suffix: "37897c14380b3e0dbb268d786f8bb5dbd7aa1b322a62bfa21e32adf609c5dbf7",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b07e7381009cf6b335e8d50be82c60de5df30bb0aa8dcb0888761129a08fa649",
    "sha256" + debug_suffix: "df4375d7c3b241f82f0f5025d0763edbe892b5a191f21bb516217d025e33728d",
  ],
  "kernels_torchao": [
    "sha256": "8aca8abf72ba772d1ed589fd709a1b43db247000984f0c33b21f399ab6b1b561",
    "sha256" + debug_suffix: "706d7f4bd46016f4bff7821a6c31d8eedfad107cc256f3e5560a9ac4135736f8",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "60b99d721ab28b8b4baa63f9dc7acacca4ba3bc93251056668e23fb3b4b25e4c",
    "sha256" + debug_suffix: "9085a7a8dece0cd339be06799a485a760821a19eb9f100e40b0323b685a0a54c",
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
