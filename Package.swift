// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260522"
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
    "sha256": "c99df2a0cb0ec1775b9874d88e8ee4ea3cd71abe7da9dc683091cf388292eefb",
    "sha256" + debug_suffix: "2579636599367425f17ac9d6978a5abbca82962a5518977027fec95cc92a1d99",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ba9cc2b608f754a971fc74168443197a22220105db4152330a34aa70dd830123",
    "sha256" + debug_suffix: "350c19ff305104aa99698f50a2d7cd8e765bed093c5f23ec05861f5a85f9afc9",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "9e8da2f84599f20de4f156181ca90d75b424ccede0ba02104f7e0139ac55c8b8",
    "sha256" + debug_suffix: "41117fecbacf6ba46b636b770339fe7ca370e1f958521d2461dbf630b2ef8f8d",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "67fad3668f4cacac4093f31b0a58c991cf47d1a919c8aaf3fec5a08f449f1083",
    "sha256" + debug_suffix: "2c1d09b46e99f4381d82bf60ca83e6762f2f03f744763a2becea6216c213cd1f",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "9488c398e237d435dbb332bc06bd7b18f51d2dd9179210e03cf3cbe7a04d732f",
    "sha256" + debug_suffix: "3a901c209ef8851072b6e45b74e384a4670872e8e5df5741f6ababcd778ba0b8",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "13270f80e0d5c28764c744e6e5c68dc37b3b807b70efcad2c993c8ae80ad22b3",
    "sha256" + debug_suffix: "d156df5a2ef7f6f355f788e44414e180a5049b81ae3fbedf3e31af646743a885",
  ],
  "kernels_optimized": [
    "sha256": "dff9274a12c06d0fdef1f3af1f36151180359616e996a66641dbfcd3cd39092a",
    "sha256" + debug_suffix: "deb5081be433edfc0d704f585d0c8ca3331256ad1365d8ea0af12b795733c951",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "8f6d53c96ef36aeb923c9cf5e3065c630459cbb6dab6c61b28f5d6861ed4bdfb",
    "sha256" + debug_suffix: "f337d0bf8edcfa961e3a1c822479de8776bc3aa19bb3d7d1757a89264c433bce",
  ],
  "kernels_torchao": [
    "sha256": "ce2e389b3d61db8d5cf4d77cbe0ffe3e1a6c298dbca476f902d86d7e31a74649",
    "sha256" + debug_suffix: "3def9c56f2b6e0132e7e1dd0ce86c4b90f9963481cddc80fab7e4f7f2a1de1a9",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "afa88e8ab787cadc5852b1734e3bbf450377f492a8f814d30a191bef99c3b45c",
    "sha256" + debug_suffix: "f8aef901ae11d1b1af62ec4d49c9da083aea4f85e57a03b83c2de0f4f94cd538",
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
