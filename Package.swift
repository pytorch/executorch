// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250320"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "5656983c5d93911e32f5aaf2ab5fa6b7d2421649435090f7e2aacd2f5aca8c27",
    "sha256" + debug: "601a0a8200cf3e6684ff288232b0b9e06afa7a8553e2e16715d25d166408d75f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9a3b1acdb54c19b3507c201024600ad4bc398108a13ea4a8252ea1e42ac6676e",
    "sha256" + debug: "6cdb80b30afeab09f596bcbae6070d7baf3d4ecb33ad652df00e2c76256c1916",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "bbc15945d3d86d8921ef0545b876970127903c3b474a6f6c367e1c2d37000afa",
    "sha256" + debug: "be2124cd1b98160ecf3aebc9e8d0519200f022ded819014f1071782ce0f5bbe8",
  ],
  "executorch": [
    "sha256": "642d89af244fd3822cbb486c738af2d4ec3f6c14eaa026edc2a012b320ad47df",
    "sha256" + debug: "e398b0e712b81fce051b752b14602e07b2ab1c009953be8d142c64e46135f72d",
  ],
  "kernels_custom": [
    "sha256": "380357e9d3613efb6024d733ec1337056db72bf719bedf8eff986a6485dd6cc5",
    "sha256" + debug: "b4ba041d432c0cdfea0ab5025f58769b4197f73aefafddf8af1e70ff86ca4a3b",
  ],
  "kernels_optimized": [
    "sha256": "4002301455afb271dc18aa0f6010dfbb92c28dbbe54ba6774345fae0acaaf44e",
    "sha256" + debug: "4151716d7cf5f9f82e08cdedf04df63a73e04121e8594dddbe9a89dc494313f8",
  ],
  "kernels_portable": [
    "sha256": "818a93a7af39bc90e3f02da5c363b84f545a1dc52bcc3adf17c5340b625ecf76",
    "sha256" + debug: "3388a1fc6b343a5b9282069de8e0af2f9d249f90071fd2ca8bc1d504e8b32afb",
  ],
  "kernels_quantized": [
    "sha256": "bc27c84cae776672a5adbb7a6457ac27654426e7caad072fd7af996409207c0b",
    "sha256" + debug: "9979a78540ebecdc0accbbb4d267961fd6f706ad5a803ed7d30404ba2ec88273",
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
