// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250110"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "b34b6cc0254214f7716d0030d6649fe27c74f94a09042a4da98bc78aaa16f08b",
    "sha256" + debug: "545af79a064acb4921309d3cdab2ece2e448f5bf1f832dc705c607e031b04fb7",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "25d3fd7c01dbd3d35d3e3c8bbe67142b8c07e10ca29a179af90b29df500e5364",
    "sha256" + debug: "008dc7f325dc04f50b06bc479dd2fa5f61210aa2a65c128d5a072f59e5b604ee",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "22d405f1fef52aa0ca8fc8e72ee69d5e92461a50534605905b2dc3ddec6ece9b",
    "sha256" + debug: "4bfae038b6926ec96317095d50c285e45867acb894318cfa50df75cac580a898",
  ],
  "executorch": [
    "sha256": "d3f2de3f99c9eee3e8acf5b146407e843fe7db622c26c013efed7c92151011d6",
    "sha256" + debug: "6c9b58dc2d6b76a6eba79064d19ef382685f187980710d2ae29e9ad5bd4f35b3",
  ],
  "kernels_custom": [
    "sha256": "d505995e60d72a9f4c716a272c9d350ba10c63f4286c166bbe3217388b1347eb",
    "sha256" + debug: "1ed78584da4ae91e06ab35225261fbf38d8fc1824365f1df669ca222d22b872f",
  ],
  "kernels_optimized": [
    "sha256": "3690fe99dc7b4ec0a5822aeb7f1bff3b8212670adff84d8b90e222932c02e13e",
    "sha256" + debug: "063eb6a474dd17a094d554d44f85ef2ba7a8a57514059e1d5c16fe7ae3797449",
  ],
  "kernels_portable": [
    "sha256": "f6835ff2fcdddec8d3dd1062c5245dcd61f9ca260085e97a4ebec7e651a8d906",
    "sha256" + debug: "c84f062d548a91c5bdb478c135ed205cc659697e7e54f6c7ef4f8707ca27fb0a",
  ],
  "kernels_quantized": [
    "sha256": "baacbf6d692c0db9875d431e174c2bf6c53c8656ca1608836527edeea53320b7",
    "sha256" + debug: "2f61d68eadea2d8e3367cf0643c81948264900575e720e66e396e019932e2c56",
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
