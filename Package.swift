// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250517"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "6e8ba16faaea38323b7a58439b98e6c26f99901233e346a3de1b2a94fbd01a1c",
    "sha256" + debug: "a4aa5da89a29c5a92cf9a186a3826944cc374307b4b23db7f5ec736d0e0ee106",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "867fba47bd5bb8c856d0402aeb1a75ab37b8464887e93a0209630ea08c2208dd",
    "sha256" + debug: "95c6a8f13f45b51fb4575cf3d94263ac9444a9d5e7756e064876ae28b4ba3001",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "582b12d0ba1059859ba737e69c1783e924d1b74acedf161c14122dab6187ca2e",
    "sha256" + debug: "84599a147629bbd58ef6b6c6b6bf248c410cd3f9f4a91f3cb7adbd853835ed78",
  ],
  "executorch": [
    "sha256": "ceb3327ff221874e483c2fcd806b0ec61cad133a1650ec8175c073934af49473",
    "sha256" + debug: "df815a007bedb164124782c605e7c79b04404e6a43b7b5046e78b765c7ca2a14",
  ],
  "kernels_custom": [
    "sha256": "d06ed1e1e50d7d01304b9b8badea882aafdb30952e6d5563657275c72742215a",
    "sha256" + debug: "707c6a51b9c1a88e579c66fd449cd3caa1e8419c8b061a9c49104676e053578f",
  ],
  "kernels_optimized": [
    "sha256": "bda325491228b3aec23339f05035d4c3c79aa2f6a079e667dfbcdb8c6bbbd323",
    "sha256" + debug: "22d60dba6ad8981ec72c3c409c18701b3970e5da7b414efd75685b365a273d05",
  ],
  "kernels_portable": [
    "sha256": "3279e62c26e6fccb022bc52451db96f7b18f9640b7b381a3c4b275d38b76607a",
    "sha256" + debug: "6048b7b835dd959286c5182a43760d011bd980fe8ae816d8b7560d7726837d46",
  ],
  "kernels_quantized": [
    "sha256": "e9efa6ecfd36555bf8efe35ce58042b8d0e09a2c07d16e5a7b2c030d76023dff",
    "sha256" + debug: "cf6542a7a4ff76b5da003fd4b2d4a5be4f678caa480a3fc575fce1d21c338f5b",
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
