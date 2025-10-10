// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251010"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug_suffix = "_debug"
let dependencies_suffix = "_with_dependencies"

func deliverables(_ dict: [String: [String: Any]]) -> [String: [String: Any]] {
  dict
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      result[key] = value
      result[key + debug_suffix] = value
    }
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      var newValue = value
      if key.hasSuffix(debug_suffix) {
        for (k, v) in value where k.hasSuffix(debug_suffix) {
          let trimmed = String(k.dropLast(debug_suffix.count))
          newValue[trimmed] = v
        }
      }
      result[key] = newValue.filter { !$0.key.hasSuffix(debug_suffix) }
    }
}

let products = deliverables([
  "backend_coreml": [
    "sha256": "d2e9e9821db2c5715e30790950e27270a93be0f2ab5f27b74a4aa94a3db202f3",
    "sha256" + debug_suffix: "9a890e330a86f8ab310c93c67c9819141f1ea0a82e51548e0eebe0f3d24afa68",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d4292b96a49430afdad0d7963aa4f00b6c7ae0a6138a85dfc58959d55c9789d1",
    "sha256" + debug_suffix: "f55276809614354d8507a6d5f62a428205d7653d8401ba694ed1d216d5249795",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6860c538b38c66a8a905d6f2f6d627287fbff4f198468e3541d1867d46c1a898",
    "sha256" + debug_suffix: "f66c65e908f929caeafea15a250b9d95ed6b46a7e6eacb4f31b5e3bea2512a15",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "075a9a122ffbef4bb6c7a2ac3c8476850951bb6d0b6bbc1138538521fb1732df",
    "sha256" + debug_suffix: "73a08643866889530578100ca29f71afcd2683c14b26903a564e11d711d5f909",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "d12fd304a06949620c0dd974b6ac2f6784041d6ea50792b96f144eea8b11ad9b",
    "sha256" + debug_suffix: "623ff658307186a0cf187a8343a2f27861916d7e1e92a3747259994a7cb35a20",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "534aa754e437c869e76c478553d60902e6f7a0ad1fbc04621dc93552cb0e006f",
    "sha256" + debug_suffix: "faa0cc4262ff4fd3e630eb505cdb28f33e7a25a2ce89e402fa377389dd520901",
  ],
  "kernels_optimized": [
    "sha256": "00cc0e304903be0ac33865fe6968b48dae178dbddaad5fc0094a6089027568a2",
    "sha256" + debug_suffix: "e324ddd7e330c0c66d1ed8d6439a3d8b8d47efed8adaa573f955b3eabcdfad5d",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "2afacb6bb57023085ccf070320db7c89beea63f1ef26f220ba6809077fecef6b",
    "sha256" + debug_suffix: "a95a6a1b9ac53c72ab055e57383feef88c37a9e920aa295e02c0710fb6107e98",
  ],
  "kernels_torchao": [
    "sha256": "af0f934639622d1423e051fb31943ee6ee7ac683d9c0cddda32d689f919d0ed9",
    "sha256" + debug_suffix: "b4179f6d08715265287d708eeaf3d5e4526a5abb03632d1bf6220b59495fdd49",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "1a8c612b4a4a8db0c01ffb0c736b3656b405f7dc00d8d496a26b13f6fa4e0c69",
    "sha256" + debug_suffix: "a1ffb120782d9f4ad7fe6caf764c818ea6ff57ca9859a3cd3f4c32710133a83f",
  ],
])

let packageProducts: [Product] = products.keys.map { key -> Product in
  .library(name: key, targets: ["\(key)\(dependencies_suffix)"])
}.sorted { $0.name < $1.name }

var packageTargets: [Target] = []

for (key, value) in targets {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
}

for (key, value) in products {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
  let target: Target = .target(
    name: "\(key)\(dependencies_suffix)",
    dependencies: ([key] + (value["targets"] as? [String] ?? []).map {
      key.hasSuffix(debug_suffix) ? $0 + debug_suffix : $0
    }).map { .target(name: $0) },
    path: ".Package.swift/\(key)",
    linkerSettings:
      (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
      (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
  )
  packageTargets.append(target)
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v12),
  ],
  products: packageProducts,
  targets: packageTargets
)
