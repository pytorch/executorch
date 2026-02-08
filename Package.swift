// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260208"
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
    "sha256": "522e68d3fcf28b85589ed48550c45474204fb4216bd07cce357a82c543ae23d5",
    "sha256" + debug_suffix: "b073ed06d1c574d1843e9c745a41fa6a7167e9c35c23055443ef41c8ecffd194",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "83b097b68457acb623625b194a4bbf32878710365806c7701ff99bb0b4a91001",
    "sha256" + debug_suffix: "a5ea12ed60aad1cb9ae26248e65a76713d04f910dce637f52bc9f32f5b515f1a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "151c14401f8a09c0f98a7f1a9818ac30d0f9eb0a2df1ba1928bb18618cab15a2",
    "sha256" + debug_suffix: "82eca263074bba084ca7eb9d4706b259a6b037a2e45a401b3adf2762219fe49a",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "2d94f1b502e166d44795269b96b1ba6345db201a0e293e076ca9190e3892cb65",
    "sha256" + debug_suffix: "953fef406b7b127a1dc69a00047dc6a9222a134909ffe8f1f30b2a9f1ef082f1",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "864ad921e4761e1a3e8e8bb654425ac82c44edceda7d7eaa520c39810e0d2a1c",
    "sha256" + debug_suffix: "f82486b023fb455f563b1939511f4291e36727076eb349dc4ce7240ec6510a67",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "914e7f148bc0ef36c165e70328095afc22a10aabbeef66925d0b76162a36c75d",
    "sha256" + debug_suffix: "ee51243a5f0903bc24abcdd95844d27286290f7e94383cc262fbb16602fb8c82",
  ],
  "kernels_optimized": [
    "sha256": "841cda142ef31ffdb93c1e0752d8a390e32e6456a47495820ee5fc881dd7d134",
    "sha256" + debug_suffix: "b6b163155bd3ef0e80a2f6abc2229664d637bcddcc41d1cab7ab9b554ed2e46f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9e8266858229ff0b82652de6ad5f06b7079cc90cda0bc04f7cdb7a05a539bf20",
    "sha256" + debug_suffix: "57ed2b64991a7cb05c8b2cfe5fee64e84a785d5807f0dfa9b18f810baca6ed7f",
  ],
  "kernels_torchao": [
    "sha256": "239c5e37099a32ff86cd3c97625dc869354a32e6e39a2e9ae1f1bfbd7b469b63",
    "sha256" + debug_suffix: "e771c77622747b32318fb35efccfd2dbd267f382a609844593001dcf27d404af",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "6b323e29faad66e0396ba5e42d9e3395290fd79ff5a3af74a6c40c67c8b3056c",
    "sha256" + debug_suffix: "867c8bfb9f9e93a9d6f695c06464104f61d61c7931d9d8156883d0c2163d7958",
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
