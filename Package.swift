// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250228"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "665d37d00a7869fb806ca0850ad315a14086ebd13509ef9c1b579486e7a9952b",
    "sha256" + debug: "68ceaa5fb269712729c05910e5d3177453a37ae65d047bf664a5903f287bbffc",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8b39adf65f6fdddb8fd4be5e4668c8c65c519127ef7b3bb7ad620f95e8bda450",
    "sha256" + debug: "832730178782a9e7d01a558ce01612a4ebdeb24c6a6237ab3e318c7c738b7605",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "fa91c3f62b5d0405a5999373e5ca6b3da6482991e60b6cefaca6c96d153e60a0",
    "sha256" + debug: "382cffe84bcb709db7674adddd0eb8fb43272de029e81769be3558060a72e619",
  ],
  "executorch": [
    "sha256": "bd97275f7f5468f4850a713b2f8acfd0abc7cdbc64ca93847e18d6b41716d6f3",
    "sha256" + debug: "1ee26b0928118967789caf9126e9a2ef6fdcc4c7ad623630d10286c616bac69a",
  ],
  "kernels_custom": [
    "sha256": "22b34f0fbf82df3b27ebcfd4c8041011efea94de5267056fc75bbca54198af8e",
    "sha256" + debug: "329503216b6d63cc1c9a7cfa1bf7137db67deb64ecad1b530262feae4f82037d",
  ],
  "kernels_optimized": [
    "sha256": "4624ae15d2c275c680d8712946ba0a77a38c6aacef6b4b753b701984ce2cac5c",
    "sha256" + debug: "25e63184cd13583a2fed832fe035bf4a7fc326dd5c03f3482b501a343771cf62",
  ],
  "kernels_portable": [
    "sha256": "20438e55f5a7dc81e895b65285a2eca0e2b053ef84a13218e8028411facd5d83",
    "sha256" + debug: "af6c5a437bc9b75b96ff7887d60b644f83014590e6d0f4ad83b2bf1fa39c890a",
  ],
  "kernels_quantized": [
    "sha256": "b07d192cfbf4817d4a024786e46185061a9a1dbe37100324faf60d78a09d1880",
    "sha256" + debug: "a8d4411900f9db1e45f704e32a0011f2c4771390ed736c75f3bd587fb0e8deca",
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
        path: ".Package.swift/\(key)",
        linkerSettings:
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
