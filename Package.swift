// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251217"
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
    "sha256": "2c8d0da0fcd03d365c65733743e9c84287b1a8c5a4f177de33aa1cfc239b33c8",
    "sha256" + debug_suffix: "083f40a277eddb06dc09b0de0363cb56b196d7f3e1d783c9f4cb72198bc2b3db",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "53121b5a2423e38bca469eb178f71a10935cadc52f1852f91faa0c8d1dc26858",
    "sha256" + debug_suffix: "98cf2bfd755b3a2692153bdfe700ffbfec3dfc8cb194fcccc4b2a676d9c21234",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2af62be5c84053a99abf8b9aba7270ff504486b5a1d63ab4d22f5522cbb6e376",
    "sha256" + debug_suffix: "6f91138cb40ce6f45caa081e4f2d55d624daec11c348b7c038014e65dd0d1218",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "b3254d44c28f5053f4a9c8bbe2218a53af8a7154a0579d7d899acfa88a5e82da",
    "sha256" + debug_suffix: "6c87d6dc4a3be82729c327bb361e6d4d6872dea029e1e792d19ed9b2dbeb3344",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "18a761a95684d9300ef08958c3400d49c393ba39e4319c266f9507a84f8af36a",
    "sha256" + debug_suffix: "f7fbd036fda8dea5c1c54ad1be04255b5116f1dec10fab6c50dee8c7cba2858f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e7e17db285536b2b2c4cb0ed1748ddea94dda11488a25a74873db2ce3624f2d5",
    "sha256" + debug_suffix: "a4e7125a4c4b59bae6e6c16dbcdb853cbae755d5e444875f40888940d47844c2",
  ],
  "kernels_optimized": [
    "sha256": "580c1655865bbb1541edd981b964ad0bb469cb962be51d51ec1e985090365a1d",
    "sha256" + debug_suffix: "54889cd5c940b12e2a802f4b336bfe907db22b9b06166710cc68dd68886cf8ba",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "052f743251428632af03c341c0b230960c3cc5032f1a64bb77782d5f34071b3c",
    "sha256" + debug_suffix: "a0acb6152c80c6142e3679fed33eef8589541ad8bbfb7e373278f10e2e1882b3",
  ],
  "kernels_torchao": [
    "sha256": "11958640a6d9898ead40bc0398937116befe4b814a24def97ca01aa0d9f38bc2",
    "sha256" + debug_suffix: "84ce8334d8179db2d4615a5eea1f71151a356e7cf34538369b42dd214439eeca",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "6a8639b90818cb99334fa98883d25da17fe4d593ff653eb1ef590a76fb21ce37",
    "sha256" + debug_suffix: "af8966a53ab3307c5c7f1d18388511b85d45d47655eba05743d3cf572f8a0d92",
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
