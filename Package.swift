// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241218"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "e9ef839c18fb2c7bdc151a4cbc19a4d4fe112cca76a0f0cef65398164c7aa5e4",
    "sha256" + debug: "2f268e6c0d0565331baefd047017f46ff77ec4fbbcc83bbb8eaafb26482012b3",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d6b91aa9bd3d91ba112c3f3918e6fec3d3223dababbb22bd5b14a0655f811d09",
    "sha256" + debug: "300af76efcc32ef679ecc539a685fddfdbc075aed104590736127a3a1705948b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "fb017447583eb29176690dcf503f99862cf730e5588e75d277a904ec79de4099",
    "sha256" + debug: "6cdc6de5dc58f3a0931d832bb4c656538712f5c0d6cf6a8f812a30bad2c4c1c0",
  ],
  "executorch": [
    "sha256": "e1e309e3dd1ea0f0b39bce42112fc635e6846271cd380a324b978b447cb5ac41",
    "sha256" + debug: "9eb426a7d3bfceb8efe879f07687631ac4b93c35c46ee6fdb54c120905616e97",
  ],
  "kernels_custom": [
    "sha256": "cfb59f55c983774a68d88089e67fa8c184eb95c3fdd9a8759fd180f32c5ee77c",
    "sha256" + debug: "93f90a05f8338d094fdcab126a248209991e7612b679cda873f2ba16df01fe1c",
  ],
  "kernels_optimized": [
    "sha256": "79784e035e31eda982bdafeca7f442092ccef641889b75d425d0f5fee0d4ef02",
    "sha256" + debug: "648d06cbc2034c9545a3df86d5f64b071695ad32a529673d8e63214bdadf53a8",
  ],
  "kernels_portable": [
    "sha256": "8a4baf0d51a50af9fcd9d6427a771c7e3b928c1ae5e91667fcbdb672387639e8",
    "sha256" + debug: "22be22291618250d2747246835ec590d9820f3cc361bf1e24e036124e55ee0c4",
  ],
  "kernels_quantized": [
    "sha256": "1ccdc38885131bfbe5e1b229e0339a3c8249aaeb742f1d5c143fa3a43f7804e9",
    "sha256" + debug: "210a5d220df95c6fce745231525d71520366435c33a0553e59cb6a2cff7f40f4",
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
