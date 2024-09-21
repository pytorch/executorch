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
    "sha256": "7083dca142f2dedf2c061d9949e6693a674b5a9ce8242f7c237b0457dcdd2ddf",
    "sha256" + debug: "77faf556d20c5ea380ba42c08df6d0f8caf26e5eb91eba7dd34fb6341abd2a4f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "975a767ce755e15328575d4562954564d5ca1fd5d566d93961edaa71b91d78b6",
    "sha256" + debug: "d3e7605df760497218ad6238a9702fe92163f32118a49b00be19f89abf815589",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5136b204da3475824392469df9cb783a075376a9634829fa1aa8d8134b41ca94",
    "sha256" + debug: "2ff1c0154964cb3d96a83d4dabd8ae422120c1adfd9e4f766e693ed37735a1da",
  ],
  "executorch": [
    "sha256": "52a10e42e7ee785ce6071348c7893e555089a81b0e87093c8919a2d03f6fad08",
    "sha256" + debug: "14309d3adf2526299a2de20fd9caa38c51a57f4739ae5366a5ad516c4457cef4",
  ],
  "kernels_custom": [
    "sha256": "b9c66a4de6b32617985aeb20872c61f7168bc1fd6ac3c567921e1595f0d107d4",
    "sha256" + debug: "99320b6a1bcca942544e092906820ef3010f5cf48f70281735443a3a24d85bfb",
  ],
  "kernels_optimized": [
    "sha256": "ae8894128ec83b468ddb69fab5642a3a7b602300c381c103fcf286d2e99bb6e4",
    "sha256" + debug: "f7a666e44185bdbb109ea08e821eb29eb06e7f2ee9843ccb5eb8ace7202b20b4",
  ],
  "kernels_portable": [
    "sha256": "c3157892731238db17f9960371ac1fb72e780c43487d7a02875549b4a171bef2",
    "sha256" + debug: "8b08b4a8a6a755106f629f095d81e7b92a3791052eeed990e70d18ff126e8fa5",
  ],
  "kernels_quantized": [
    "sha256": "5444ab9e47ae28721f57ddd040a0213076a3b77a8b6728618582c16ff6171f10",
    "sha256" + debug: "9547e5e077f0927854d45b140ddc4d7d8facf251b612421b186671c2275bb192",
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
