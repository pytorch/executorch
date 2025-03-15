// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250315"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "33b935810d0c91a32fbec257b76ccfa81262c36637f763230c9e4d0d55fe7f9c",
    "sha256" + debug: "7d2be3d432924cf066ca444f7231a38f5808bc3412e33767f0969c518254eabb",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b870c8b8747f2fc680bb210bd0ceb56015908cfeebcedf84c846ec957050cdfe",
    "sha256" + debug: "aea2210bd6b71fd85718f7d110fa2a88233843dd320bb9e8a6e2ad3f0f5b1767",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3f6714a5c4172cc0d378a0837c010b6b2323a95aa15b3ea0c4a28748fc5e1f96",
    "sha256" + debug: "cd71d06cd98156958592d508979a17d08423ad9be69893ff59afd85a5c98b948",
  ],
  "executorch": [
    "sha256": "828884a5c662ef5a6f2c6140249f3a1ad1656aec4bb29c8b21d3377545c10680",
    "sha256" + debug: "836a6d7c6b075dbf4d8383460e579e54b8446c153b4be89abaf3061e677f1ffa",
  ],
  "kernels_custom": [
    "sha256": "aac89c981cc05a6627011f09dc4e112b440c480ee04823564fae16a4577d13b5",
    "sha256" + debug: "fbb3986cd10671949a7517a4d9b756ebe4e0f228c0df99b2b9a79680082f4818",
  ],
  "kernels_optimized": [
    "sha256": "c94510ae2680f806c9634268262ba7551386a241c70015d59ddfc55ebc906ed5",
    "sha256" + debug: "15456199fe281df2c451d57f0046fa8292e06403262a3e812fbba217efb585cb",
  ],
  "kernels_portable": [
    "sha256": "d38a69eaf05e140431e28ef40ea6f21394914eb3eebb4780ce3924191a5159d8",
    "sha256" + debug: "a33f1e5df128c4557b2a005671f4ecfd0325da8c934f967ac5df769f10c8b480",
  ],
  "kernels_quantized": [
    "sha256": "54df2602549938975ee47982d3f2bfd76b36f440d1980b8528560b73f6201415",
    "sha256" + debug: "49d683e507446a501711f32b7e9b6a12475a54721cf7f1ac50557e124ead3ea3",
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
