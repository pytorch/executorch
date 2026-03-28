// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260328"
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
    "sha256": "adba6b9a1110652235b00fa39a1a07b048ca042b32cd72ed1def6a0c5b6eda7c",
    "sha256" + debug_suffix: "a2bf9a6c744ca985d23b46b9b655bf59d529b6e9efe4d63aecfc575343788a6c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ab13d71f776972238decd3595d4e3e5b94ba2dbd846ae63d65beecac21bbfca0",
    "sha256" + debug_suffix: "481ea01c33a0a9d99542dd657a825c5bd3a581d6f2f822df2aeb4d0b30c173f1",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3a433568a1646cddfbda7093815b9d9ff2af4a74ade232babf27d1f5debcb9d8",
    "sha256" + debug_suffix: "db3bfbbab065a5f6d336b77dfd50d8b3383fcf74e107009159bf08306bd4563c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "64968871e7b9d821a485e4faec6d4495115e3c4191ad01ffe79a5a59526d5f1e",
    "sha256" + debug_suffix: "2227f8ab54aeb153e80afb679e96a7c621c7bd8a029be2e3d7dd510f07fe49ca",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "2e910b86098babfd6f50b1677eb20dbf076e08c1eed708e0ea543ee4e4b4c139",
    "sha256" + debug_suffix: "5d69622e73ee949ef11eeb98aa3b27325b6b13ea2ee6c9c78aa5c5e5b57d8eb5",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "44deea373e18aefbb6e05c866d31993360a4140209898cc0400b2a07b78fdced",
    "sha256" + debug_suffix: "35d66ac9732ff25bac51af147e2a7fdc664936ed94454e059c04ded9ba4a2201",
  ],
  "kernels_optimized": [
    "sha256": "c404f76f87019303d82dccab3546229caf22650e01e259d7e886f49cb57f00ca",
    "sha256" + debug_suffix: "df2fba1e20461726f536cabfadaedff8771dd1f080178f22f8d3b9568a6f953a",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "99f2d4d560853c0cbf1dc4efbf6ad04cf84bb0bbc5e4db1e50fe1b5b86de436b",
    "sha256" + debug_suffix: "1af93ef22d11e64e4cccb2b3a341e03076cae7316344408d89f8ef240415ad83",
  ],
  "kernels_torchao": [
    "sha256": "6d1a44a277537cd1ba53139e792353a8aaf2fee774c947aabdf8b4842e76210e",
    "sha256" + debug_suffix: "5ebf08df7dcceb86b1518c4f5a0a9e3a7e8b591f91a6e302ef8c4e7e0cb04c1c",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "425e6cbea6326885d85cda53bb1f1dd26ab94de8211860df186179cb0fbd8ebc",
    "sha256" + debug_suffix: "d4ac47f463a85a60d33902332da813ccff07cbe1c43e4db7ec4e9dbda50a08bf",
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
