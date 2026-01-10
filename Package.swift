// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260110"
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
    "sha256": "4e0f51fd2c403a13299a1049d64bf631fd5a34fe90bc3e818f35ff208aa9d6f4",
    "sha256" + debug_suffix: "6223eb330b1313e6a7d309d83900fcf6e938f3c5d687a77582499e9a2477b005",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2896a525f084fd4b928f483c897d605166a5236ca7ab1fd59b652add5916b2f1",
    "sha256" + debug_suffix: "5f36ac0209750ceb40fd281cfcd10f03264d608eb7dd0407e545850ce320b493",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "72abfcd190ae0e9257f941349c4d726da2a4cee3ff4a8a2eeae669f819f9e532",
    "sha256" + debug_suffix: "d0d396f3e5b4b5092a2051d5429287670b42ea5180c1ea394c89e6bb79f0b792",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9f4998cdfd101fe5c14bdc9f028c9e79a7919ba6324a121540abd7c8a3f8e59d",
    "sha256" + debug_suffix: "e3ef7d159ffc8bf6ede2e12c6d532c7399030f24c0daa7d66b4edf7bdbc169a4",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "d1e89b5636490c0e7ffe9de7a2566b2a54fda865b94d6ee0c864b36f4513f75d",
    "sha256" + debug_suffix: "bcc89247f7cd339cf664caa2bf578358c93b3f9e380a297fd11e92b46bea3ba9",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "93391f09c3bbbeba3b1ce5422f8ac6376477c9e2eff569ff79084f0b9ece7028",
    "sha256" + debug_suffix: "2010445133bbb3963a836ba9ee25507a16252e10d00d73a03254d4a78623dfa1",
  ],
  "kernels_optimized": [
    "sha256": "de63b9943a0b9042122a2144b01dfaa2fb8ffcb1cc4bb7ce6ee610d205cd92a0",
    "sha256" + debug_suffix: "96dd2b752757a82ad29a51b3acadda76181edc85336e81f1383b37783bb6aa03",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "fbe1aea1c30c858fcf2d8d2759a112ede818f415dc09e09f5ec917dfd29d5d9c",
    "sha256" + debug_suffix: "e3b06626341d68c4bf856c7b57b148e19c9ed71e024e91fab138469607019a3b",
  ],
  "kernels_torchao": [
    "sha256": "307edcce619f92b07d9e350ffe14111722c0a5a0ae6f9ec44675571c33cf2a82",
    "sha256" + debug_suffix: "b7a41db8024544ad57662b59b82fb3bde950be2c3dafc77f1a1704275f2e8102",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "92eff8f9becc8d4a02190eee2b323c63e334b4f99ffb8f3ed984ab9221d265a4",
    "sha256" + debug_suffix: "0c9c0bea525142ec0f6f2f4dfcd7bb8f449fd7d0240e099786a16080aa70e5f5",
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
