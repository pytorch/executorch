// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.2.0"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "coreml_backend": [
    "sha256": "9471702164d84d3b4f8da96cac796d9a2474d243175b25638e82546ca7786b77",
    "sha256" + debug: "4597f06bb91eab7014a959e060cccf4e7f9ffdfe2f62d548ef6589cb797824b9",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "custom_backend": [
    "sha256": "451917bd432fb66494d29d55711fdf389134c2953bdafa3cfc2c36ddc62b685f",
    "sha256" + debug: "028cf558b713b49bc4f5f4ee816d43818f2d6a416be94da0f8a256d3ad937202",
  ],
  "executorch": [
    "sha256": "a693533614462b24dc175c72a7b5f5c2c0146eceb7339f173c06b914b6a3e69d",
    "sha256" + debug: "c043675c5b19347af6299413aa27c8bf06877d49254e0aece9b531244b068ecc",
  ],
  "mps_backend": [
    "sha256": "03f8ffdd1b4bc1c365c26edb948f1c2e12ad7cb41df9b8b8441e89857421d353",
    "sha256" + debug: "4141537aa4791d8245924704dbba3d6b8dacde74e0cf9ebbe0b78b780861d6ed",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "optimized_backend": [
    "sha256": "8f050023869eda1fc610c344151df0ca99a02a3eeeb4a8cb2b017c0fd1784559",
    "sha256" + debug: "e98e75ca84982f0aee1182177a0878b3f93e373d6189e9a776c8ddd2354f8f78",
  ],
  "portable_backend": [
    "sha256": "dcf0a1c5cf5a6a07ba35d96958dfa5b89d31e5984fa8ec24316ebae4f9eb63bb",
    "sha256" + debug: "eb8b4407d8fbf9b8489be2de9df2b701681a675e31ce708185d2e9d9e3562c4d",
  ],
  "quantized_backend": [
    "sha256": "d23c52bb6f3d40e0f6bf54e39e63c535f328a71a11cfae90c6d7f909ba159b8f",
    "sha256" + debug: "d779da85424b8f7491450fb36d4f1d646c48e131b66e9d5f6f9f81c4de9c1237",
  ],
  "xnnpack_backend": [
    "sha256": "2ccb6d4f3fba528d5abd1076580ff7b41f3263d968fa9ce6ae73b1e564e15b97",
    "sha256" + debug: "0227c730e33e9a51ebd46fb59bb3997403ec2724f0bd8cc7e7e8308b5d2c50b7",
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
    .iOS(.v15),
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
