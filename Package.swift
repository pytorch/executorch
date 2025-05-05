// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250505"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "0c37d71c246f7006b87959c1ee17b34e48ff0771d4729a8788a72a44f020a415",
    "sha256" + debug: "289e3a09111b3435c4115a4dfc8ff76763036f38ff399fff9cf0318b5f4635e2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "62887d54bfef2cb6cfbf7bcb454695c317da2d98def6315618ebf92bcd609917",
    "sha256" + debug: "aa1e97a311e86d7ce8cae33e4447358a24917f4cced04d157c62a301db129798",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "57f7f9be0b6b6a28bbf8b24d40bf46e50793b4f1751fba2b3da2d78da33a3d47",
    "sha256" + debug: "1f5291d69b00476297932646c7864d14e56800e7b5450e5678346fffe1a9f754",
  ],
  "executorch": [
    "sha256": "72892c30a329e97cfd3ac04bc45e243c934257ce39a6038ae91ceaf5726b5cae",
    "sha256" + debug: "18b75e140f6c707349079a07ae6456f667089009706905dff36f2eaded8924aa",
  ],
  "kernels_custom": [
    "sha256": "d22615ba563d6272acfd3bd82cfe3648a2234b62119435d70e3ea455f8c5f10d",
    "sha256" + debug: "ab3657b7fae251287d80a15060d73b756458448597da2107e54f29f93b2cc004",
  ],
  "kernels_optimized": [
    "sha256": "b760f6fa042afcd2d0e87573eafe34a04b1ca1ad50360fac1dd94d8b26538c4b",
    "sha256" + debug: "c2523c4216f97bf7d03eb64176797c2c44a0abe8b6eba8df1e5b39d33f87fe70",
  ],
  "kernels_portable": [
    "sha256": "bebfb2c6b4c4bb5a72b952e11db2171a83c95b46091ad78cf61a36d1fcb00004",
    "sha256" + debug: "a226faec5213fb8f9b34e01f64e9d7454f845273ae6747deb56dca6ef8b1ad0d",
  ],
  "kernels_quantized": [
    "sha256": "1201a0e1dfd532acdd406df29304dab75c7a3d44de881e1388633cda9755cf03",
    "sha256" + debug: "b2bab8cbd66734be6956b95125aca1512497ce06d85f257918ecdf1593761a1f",
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
        linkerSettings: [
          .linkedLibrary("c++")
        ] +
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
