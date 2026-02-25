// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260225"
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
    "sha256": "83dc3e287ebcd8b33ac39c0f9126da98e4c2dd97b81defb93b7cfe02a06685f0",
    "sha256" + debug_suffix: "45f06a08f441a012b722bfd61ffdd974c62a56b72126b2153f91fd39a641f7da",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6e18fccbe32139d0f94156c5ef670c111c73a76097ae9819be5fceb762ee3921",
    "sha256" + debug_suffix: "5a63acd939363746ae60c70d5e259a7309594eba4b1307c5a2c8ac7479e93dbd",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2e1ca1dcd7f57c40519ea8ef61204aa2f9a7fe825c6069851bafeb2c13ea4af3",
    "sha256" + debug_suffix: "d770d78348232b6567b066fb45c2974ac61a125895779f7b5a5306e01d90fb11",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f297f776ae1fd2bb17baedeb75e5f3f6608e0ed0ba635062af1b23a09846b514",
    "sha256" + debug_suffix: "2e4de14ed4261b3024d10dd6cca26f3368f4c3bf9cc37a0f1b844a8e02e8cd48",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "94f2ffba388ccd1f24947906fed6390330dd90278d86a01aa1ad67ff191fd8f8",
    "sha256" + debug_suffix: "824078b71eeb974576f156e307359e4f9ed19e9c57c52d7201fd75251bc6c097",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "caba9e08ad53ee1dba5184da90104debbafa4bc379fc0151102baff86f0d9994",
    "sha256" + debug_suffix: "8f3e63ad0e8956703b10c98b960658d4a2540921a7dfd7c666fb675eaa4e8fe2",
  ],
  "kernels_optimized": [
    "sha256": "1b35da56e005b51342c1f3a7a11af05658293ecc6530d001241eb75e1ddfbad9",
    "sha256" + debug_suffix: "7a418f9a5828d47e302b110cf68aa41ddadb43a45f906a54b7e7501749e64dd1",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "17ccd2f8092dc83cbbd9917aa9d7525a8a6a6eddb8b0a7a450c56d9a5e2e29d6",
    "sha256" + debug_suffix: "5bc4f954baea76dcdad49de5e94d614b4f691613b00f74286504f6719c49844c",
  ],
  "kernels_torchao": [
    "sha256": "b826b9eca8d7e1495e145232320410e2d73c2bace39564a51acc2d607bd80e93",
    "sha256" + debug_suffix: "2ee695d4e030de289c145c5764893d5d6140e87a39ace85e2b3274f58009b239",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e0cc1a4cf49f1341e4470afab9b2ac1697158acedc5495551e6e669b216b4a6e",
    "sha256" + debug_suffix: "cf26753fcb69362cf7f6cbabd836bb7e9833804daf39d52d9dc6f329d21cfbac",
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
