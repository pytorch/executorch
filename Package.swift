// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261005"
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
    "sha256": "2e5611f0a4395cef7f2dcc8ae370c95bd0a548af70248cccb59a9caff29a60e0",
    "sha256" + debug_suffix: "6fb6f496f789ff4e40428c775453049d8e3f13030fe08d75371761d2d01812db",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "90a3aa7f4afd7b7b61e6a817b303738b13bab82043642d3fcbdc8a1df913ae35",
    "sha256" + debug_suffix: "3783baef38b6d45f8d640d9572f2a4762fd8088527089e974b7664eb3ce67841",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "84d1ab177d03baf5294fe8fc6f3e5f1d1c16d0e87f50d9077a100f464057acde",
    "sha256" + debug_suffix: "c4b61707dcae27ffeb5dff3ce3525357b4ed7ac3d40d4431eb4fa328a0acabd9",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e9f32f6b5e0bb43e0a8ebd4422f07b766aa841227a0362a584900707594a10af",
    "sha256" + debug_suffix: "5f5b4ff113ef8323d5191f561f40642ed34d0f29852bf3f620d3acd8ea96ce3b",
    "frameworks": [
      "Accelerate",
      "CoreGraphics",
      "CoreImage",
      "CoreVideo",
      "Foundation",
    ],
    "libraries": [
      "c++",
    ],
  ],
  "executorch_dump": [
    "sha256": "203f30c814372847a4217e739ee51da61d70e732e5d854c20d5919645217b9f2",
    "sha256" + debug_suffix: "0c2b74201bd89b335be37fe652bac1ad31850bd8bf9e8b7620cc9e4a10a36f1a",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "73475594beab4dc7e77cad58605a26a3826b1355c8f80c3e2eaab202f805e5d4",
    "sha256" + debug_suffix: "73d9948cee2571228917f02c335de0337c512d18608a6b9547d087eb035c10cf",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "60c794dc2a5a25399f601b766c5b20cdbfdfbd9c66979d8dd65e0c293449b7c0",
    "sha256" + debug_suffix: "6e09b6ba0429c1bda4f8f6b03bc89b3f7db7fb4eb70a8101c0c9a27352938630",
  ],
  "kernels_optimized": [
    "sha256": "1dc8882442ffd5a1b164c1b6c0122e87f0056a97a0b7622776096690da24073c",
    "sha256" + debug_suffix: "c23bf55b57370cad2ae3bdb6cbf5ce6d42eeb9ffc84112ca69ca0ee74381241d",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "812bf847d74e9dacd239de92ffd3c2d9feec2400268d2e0eb657396679b71f29",
    "sha256" + debug_suffix: "fc3158968149339a03e1a1118ae4432f7091e2f8bd6c657cd4b9f651e6d523b1",
  ],
  "kernels_torchao": [
    "sha256": "80d51c36ec8ee32a8ecf5b8df454d27ad5fd648dc35ad6792ea31487f36a4c60",
    "sha256" + debug_suffix: "825899b234c1bbf1d99e5f1aa31d1dd17079b30511fa9f862279cce787ce1334",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "0d12d47d9fa9b2cf7a87679d1e31a4b85ac3a35853ef20bb352b8a39f5b31f0b",
    "sha256" + debug_suffix: "21e07b1c1a2e9a553e2d03827755130102905ce814e7df5964e55fa108e0ae0b",
  ],
  "threadpool": [
    "sha256": "e36c4f4ca718a41b77d027a496f8d0c6afa2360e1e4447558dc114186775ea85",
    "sha256" + debug_suffix: "47160d8e52091ef63189757076a04ed4e3afff0d7460b3350028efdec58fba70",
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

// The MLX Metal kernel libraries, one per platform slice, shipped as a single
// resource bundle both MLX products share. Kept out of the generic loop above so
// there is one bundle (executorch_backend_mlx_resources.bundle) rather than a
// separate debug copy, and so the release and debug delegates resolve the same
// name. Each slice's MLX binary asks for its own mlx-<slice>.metallib.
//
// The release job commits all three files before publishing, so they are declared
// unconditionally. A missing one is only reported at package-resolution time and
// does not fail the build, so the release job has to assert they arrived.
let mlxMetallibSlices = ["mlx-ios", "mlx-ios-simulator", "mlx-macos"]
if products.keys.contains("backend_mlx") {
  packageTargets.append(.target(
    name: "backend_mlx_resources",
    path: ".Package.swift/backend_mlx_resources",
    resources: mlxMetallibSlices.map { .copy("\($0).metallib") }
  ))
  for suffix in ["", debug_suffix] {
    if let index = packageTargets.firstIndex(where: {
      $0.name == "backend_mlx\(suffix)\(dependencies_suffix)"
    }) {
      packageTargets[index].dependencies.append(.target(name: "backend_mlx_resources"))
    }
  }
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v14),
  ],
  products: packageProducts,
  targets: packageTargets
)
