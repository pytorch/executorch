// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "latest"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "f420ac7d442063e1c2392fcf9d1cc5e50559c3df2e8539ac29b057c13eef0fe3",
    "sha256" + debug: "06ca2099c5bbf8305a6eb3ec7c18525d5cfd8ffd57ff7dfb0aa6171429c39007",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "36d2c4383fe035abc38526a78d05a4162d5158de6c06a8cf6ccb42b2aababac2",
    "sha256" + debug: "dcea46f167d08f3cc5b3367fa1a6df18f27b409fc28bcf686fc8e6266b718ffd",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a0b04d69a1f7e8df4a3a66e1774d1874681ca42298f5e12e46481de4c8d6a8a5",
    "sha256" + debug: "87dc32eccaf304804680eb6d60949d6a808edea04ea35f24cd5d908a8ee2955e",
  ],
  "executorch": [
    "sha256": "17905f0559fc35b69ad40a2da28fe4b3c4c047c8a67f97c290b66cfca0bca88c",
    "sha256" + debug: "eaabb8f929e9216ad708eefa279f018ffcc43dc51f740f3bdcb269f10f8a3c92",
  ],
  "kernels_custom": [
    "sha256": "40c4f9d118bb9f051fe433a189cd5079931121ac8cf8993310d584264aeff67a",
    "sha256" + debug: "f45ff25e168dcb5f3594abc66637e14f326705aae0c8a145420c0cf17979d6dc",
  ],
  "kernels_optimized": [
    "sha256": "832d718708e1a09b93fbe9951ab5d0f010964dbdee32a440c24252cb17daa578",
    "sha256" + debug: "6a0210a663d293e3177090e9213b67dea5821fa6f9529c7ed6a8380d527903c2",
  ],
  "kernels_portable": [
    "sha256": "d7d9342b871d97824f8f102c0e14633a010f15624005acd6823608a872bf8d22",
    "sha256" + debug: "670959083a9e103744ed1244448820575e72ccb9fdbd9d57e2ad0b92371cbd5c",
  ],
  "kernels_quantized": [
    "sha256": "408147849497682ab1a1550a9f5e4557e5a1ee1a81dfe6e4b3a1ca0b9328b4ea",
    "sha256" + debug: "92ebc2978ff2b606d0436d0b86c81101a770b2bc1b1fbc64e7f93f8dd5ced2ad",
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
