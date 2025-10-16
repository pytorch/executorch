// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251016"
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
    "sha256": "8546505bc0d8b33dcee30cfc38d458595e68a74650a2d3c879f5a0c131097d49",
    "sha256" + debug_suffix: "837f785342f37f2e8ed0cc7fcf810129424b408ed702565be4dc234a80e66525",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "df6369bc519b6580fb11986c44fb554dd2992fccbc06fe5322cc861404617337",
    "sha256" + debug_suffix: "47b5cbe4fdf4fecb146c3e1b66dde87acbec71481f74413f89a1a776614ccc18",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "22aebe411f8064994c10107d54028e629d641da67956ca859ae98daf897701c0",
    "sha256" + debug_suffix: "15f06a8a009b151b1b7771611b0928d4b540fb7bf63ead9dcd33811f84a29451",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "849e251273cea4524be7b32676f2392909e1b62b26258b752df5a73a052fa585",
    "sha256" + debug_suffix: "4013357ed2c3d37289675cd5068d9f6a744f7a561e4d6cd48023a4b650a62505",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "1b2aaca22e7bfa59289cefef58fd80580b11ce5f78a1aa4701083ab9a0477cbf",
    "sha256" + debug_suffix: "6a47c0c5b8c1c82cd1a384a6277e0fab487539eb184a6c2462a593f38f7e5dac",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "4ff15c6f03b8f656b091a42cb40e20ebbc7df1a8ae93267816f920f7b15ef51f",
    "sha256" + debug_suffix: "64c0d231f1d8dc08f6e0ee48ff6ceb23e6479d9167a815309f898a9a137d85f9",
  ],
  "kernels_optimized": [
    "sha256": "4d79966d92b0406161fb4fc0b40cb786415f0896e05e06af198b1802dca91f79",
    "sha256" + debug_suffix: "3bb13ef3c0a7c5afbe6e29cd9bf2f4860789444bc8f1c8f73eb79a4220acda7f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "6310ff764cad508577947977159917e34ad296476f2daf47b20a2a5504d99224",
    "sha256" + debug_suffix: "af0155548fc7b213db96d4e607214474c92055fb5d0f2bd0d672c9ce8b43cbf5",
  ],
  "kernels_torchao": [
    "sha256": "c865c687b00e5c19f9a4ce1e360c2f592d97d030cf73dec890d7b01885632bad",
    "sha256" + debug_suffix: "fb347f8dc7760acdb7ca606839a66a4ee4b721ac2ad5eb9a22ab197bf74d4151",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "fff60daecd2cb1d570f41e38257f55520e502a05f6e797018b947789e98f96b3",
    "sha256" + debug_suffix: "7dbaacea86cdd71e5d97dbaddbef3204a5d1d98fe46f95e813b30b81d75b67da",
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
