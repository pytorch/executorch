// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250227"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "543d1bad80dcf70490e4d0b946ee0b9ca85dcea1600033143efd0692e6497a86",
    "sha256" + debug: "802472fea8993781dff4b56a01079623cc1abde340eeae88daad5fd5e58995af",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "438739025aa28f9b7d39775d94fd0bc3720bcc6934063c829348e545caf41477",
    "sha256" + debug: "839d0ef5edf0a5e766f3c2e43e19cf04205331bbe4cbde22b0243ba085943904",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b8974e4763761522fe49f0fcc2cfa704f706b148b06cc42c2c18939d538e35a2",
    "sha256" + debug: "3fe35572c31fb510993efce759f48ccb2d95ebc62ef3f480d155eb180eafdf66",
  ],
  "executorch": [
    "sha256": "08f1d55bfccd270d5122fcc6585af9bd5ef24de8842f453a651881e8a6cbda48",
    "sha256" + debug: "99d07bbdd709f355752d7a88dc0900c4a114969475b4181e3dd697e2a43a7ff7",
  ],
  "kernels_custom": [
    "sha256": "784e0abe367fc895a6263e706dbbc8d3d9654ab23b026c401444a4ce3c63b511",
    "sha256" + debug: "510d5de7920a0cf11783bd5fc58b6c48eb3ebe067478588b536dd1230723bdb4",
  ],
  "kernels_optimized": [
    "sha256": "2993e15cccc3cd6da36ed7a508de13d4ee167a741ec7567ef3eab4bd4465734a",
    "sha256" + debug: "2952e58934ae15b05fc8656441c4e2ae4f7089b98f8ae6933ad95f1afaba2141",
  ],
  "kernels_portable": [
    "sha256": "cf42465698c2f6e0c129b5580542e22629c269795f77bf9413c8bdf427935ab9",
    "sha256" + debug: "b68b4cd6bd7321084fccf295e1ebeeca7fdcb234fe9d889e657649173a47f854",
  ],
  "kernels_quantized": [
    "sha256": "d9e0ea07ad0e9310125b2d47e5bce7f9073982424b21d13c75a2d63a0ca15523",
    "sha256" + debug: "7f789a1adc9331a78132e85868124e6169834425dbbfa4143efab64feec678df",
  ],
].reduce(into: [String: [String: Any]]()) {
  $0[$1.key] = $1.value
  $0[$1.key + debug] = $1.value
}
.reduce(into: [String: [String: Any]]()) {
  var newValue = $1.value
  if $1.key.hasSuffix(debug) {
    $1.value.forEach { key, value in
      if key.hasSuffix(debug) {
        newValue[String(key.dropLast(debug.count))] = value
      }
    }
  }
  $0[$1.key] = newValue.filter { key, _ in !key.hasSuffix(debug) }
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v10_15),
  ],
  products: deliverables.keys.map { key in
    .library(name: key, targets: ["\(key)_dependencies"])
  }.sorted { $0.name < $1.name },
  targets: deliverables.flatMap { key, value -> [Target] in
    [
      .binaryTarget(
        name: key,
        url: "\(url)\(key)-\(version).zip",
        checksum: value["sha256"] as? String ?? ""
      ),
      .target(
        name: "\(key)_dependencies",
        dependencies: [.target(name: key)],
        path: ".swift/\(key)",
        linkerSettings:
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
