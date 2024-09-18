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
    "sha256": "88eb45c374bd1ce5eb7f29018872b1462c8b6ae64a744d42afc05206b270a45b",
    "sha256" + debug: "a27698d84da538611357b8cf8acee5d9fe54936696f66f7d51af2d074cc38750",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "84c189fa3b8baa4f5044299aac1da1f8a9cef79ddc7b39545e1f6c67cef08be6",
    "sha256" + debug: "6c8615f9b6f9cc37a2acd64f50590170fefaf2c32d353bb89d7b1a75dce2866c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f3906927a1d7e1df45522a295ae97d9676043fa7bf6d763b60c51bbaa6510714",
    "sha256" + debug: "c518369e9cf97e0ffc7ef36498793d813ce8ca559b6889f469c309061e046b08",
  ],
  "executorch": [
    "sha256": "3e49bedcb77002c814c52e20bbf2bacf7108482e5edae7e053d4096fd43b2e54",
    "sha256" + debug: "e80eefb705ad1670b41dcbd8068074b8437556c461270bc752ddec90c67c8d1a",
  ],
  "kernels_custom": [
    "sha256": "36d12432259947b1ccc052761ccb1808f95d582e66a5a5b41b5528b40a62f12b",
    "sha256" + debug: "25242fd14c8bc9ab5e99f8f6aa2d0ab1d2b9210ce5127c685e49529bf5ef50a7",
  ],
  "kernels_optimized": [
    "sha256": "89f34a7106ffc33c7308bbeb4848b556e2a2b33995d0242760704f51e34ee35d",
    "sha256" + debug: "4e6cbb39595b5a9ccd192bdf9d04582c78a66bf7180c2e0d4ae35502efa174a6",
  ],
  "kernels_portable": [
    "sha256": "cc8390d0889eff9497bfb0e5be6a1f42e49a3a6e281f5f1e34792911b10ce684",
    "sha256" + debug: "af8dd8b35ce4a999174013cbb4763879c47f9c575b90811523e56578afb66855",
  ],
  "kernels_quantized": [
    "sha256": "8eb61927242cd25f67912b5c7f851a2deaec854f789bf8fcdcfa15a3879f3561",
    "sha256" + debug: "b1460de8fda4d10ccd5b3ca1021814c3149a98480f3dc4249ba5cf2d9d9438b2",
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
