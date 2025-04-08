// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250408"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "95cdfb5eec29e846b69f39bad88e1f625d4cb01792170d6f61379e5cabfad529",
    "sha256" + debug: "76884f193b460c85e211bc8ebcd1d5029701a94b73b3ce70a839fff314e057b9",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ef12a5ce21b0d049b3434e960ac08c984161b9f36ce940473463fe695d062497",
    "sha256" + debug: "ff9e34603d4ddee5298e30e05c95123d0b9d1950d04d0542b5726b6560758474",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d9eb1852ab6a88a9db87d801bdf7bf477786b4e0558863971d4115c17622318d",
    "sha256" + debug: "55f7debd9e717534ad655eec97d48ba878fa257195f37e6a8b1ba496f9365bac",
  ],
  "executorch": [
    "sha256": "752dca47f630a8e3431350c1dbbf641608cae4e31fc09d9fd6574d3036999641",
    "sha256" + debug: "740bdce1f6523908b023c7a3f15372f0052b420c97bc1b804e1a03dfbdc3aabf",
  ],
  "kernels_custom": [
    "sha256": "98e6f6d5700f92b934034b5c1c02664696b721b214ebda8e8078c5d29a5af29f",
    "sha256" + debug: "fd480414f86f097eb16838997ca7585d75f46e6f2cdd09554ddffccda847ae95",
  ],
  "kernels_optimized": [
    "sha256": "5ab8187894e16ec92cdc3880008723af7bdd086904f8a0b1bcfe44a4fe752838",
    "sha256" + debug: "b3be0321c32b004ed3db9e3a740381684e3be1374c0d8a5cd0f5388021d1e3e7",
  ],
  "kernels_portable": [
    "sha256": "d7edefab9be03c1e44ecc2163048311809ce60d9585c84380f242f59a0c1eeb4",
    "sha256" + debug: "f07412b3bd1db6cb8bf0016e69b45e4358d0002154077073571f12e970e69268",
  ],
  "kernels_quantized": [
    "sha256": "483e36a5c7f11162bee4d01e73a4a9836161c7868881a7702c92529e6077b77d",
    "sha256" + debug: "f59fdec5aa965fe5c5420aa101c9d92d9727c86e8d7293fc4d7d13e0befb78d3",
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
