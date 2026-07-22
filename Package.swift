// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260722"
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
    "sha256": "cbf988a5fa97b6711284b78a6bc4c4059f0a9a12a0969b9a47a693574ce0126e",
    "sha256" + debug_suffix: "a83eb9d5f86e77af80bd4a20e2601aa53ef2ca896ae9b7576e7cdbfc6ff5fcf6",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "12dd20bb71e6f4ec98ec3ebd48be1ca6595f04eacbeeaa682b168ecf41a8a022",
    "sha256" + debug_suffix: "cbd6a3ccec7f91ea896b3a15c0202d5ce92fd8618711bb7c86ea3d10ee9dec7f",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e93775008dd1c742b9cf3bf1f4514dfae531c1c43551508bcf28e427469e10ea",
    "sha256" + debug_suffix: "7b0296e5e6654a32113a341554ac0d41c5dc44a888b99152ddf95a533cad5efb",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f26357e186a3cff5d397c6c7b4398d2877933c935ee4957dfc0b03d75e40ee84",
    "sha256" + debug_suffix: "45ad0419e3ff1f02b492cc735348ae011b76dd45a1707219aac4c339a71e47e5",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "5714ef42a39da496ede80a5fbe3a543d0f5366654ceffeea5b362b338a7e8ed6",
    "sha256" + debug_suffix: "59a8e47c17f85b803bc298042b4ad6466fef7094c0b96e075d7ff50f09ec8da6",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "7b9ccdb2bb4c17be81ed02567522482b1463e2663794cd946fdd2d8cb568b437",
    "sha256" + debug_suffix: "fb7f095e2d3c5f90cdb01e5124eaf913adb90824c19c5b46d95330a4f86d9011",
  ],
  "kernels_optimized": [
    "sha256": "2f087e525f3fdc4c33ead7b4fc9eb3d26cc276e7ec2792d91b947f13bef29d75",
    "sha256" + debug_suffix: "f95557f855891681b94949083aad474a64104a1392192dfba27bbcf4f7b01044",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "851db379186ed4a143d752d7efea2404cb7cad9e6c9212490c7ffc372f9dff77",
    "sha256" + debug_suffix: "c85bc43a9d04f0910c8ed15a4ccb81569278659db023e74424e078751855c370",
  ],
  "kernels_torchao": [
    "sha256": "d152f461a06667b8965257dfb7a4b692eff541e144046ebb4c5bdac5c0f49186",
    "sha256" + debug_suffix: "3d71ec8cdb62734b432beb35842fa56206270f83671fad883b57f4e36e8f2985",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "60fc148f5fabe7f522513868a0e6f830894db536146784dad576cfe938a454e7",
    "sha256" + debug_suffix: "4b76babdec7d0b68c2fe76f07643eeae207ccca50982f60f5376683b571b7626",
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
