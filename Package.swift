// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261023"
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
    "sha256": "94d54635e56127801e46737d4cd2097bd7352a1061d448370b5d5f017ce1ca71",
    "sha256" + debug_suffix: "28fb84c918b272b72ca853f491c3e6df5b986c061b9411e41e50f5aea597f758",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "03dbbfa32bac20fc321cfc5d93e3faa7ee2fbc42e789300df56b08e461a3b00a",
    "sha256" + debug_suffix: "0eb28f106eedabeca27fde0f4a5c5a31837b178e45cf548fcc11aa8db9d1d24b",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "270f3e47f51f5bd70dafbd77946aa06ddc7345da616931b80a150fef3f473c48",
    "sha256" + debug_suffix: "79619afb58b8f08677662ba18b96835dcd201343f2f032c133d3f86c921ca4a8",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "c2d8a4a37376c5d636de2b2e3a0830fae8a12c925983af508bd270b5e8a5498f",
    "sha256" + debug_suffix: "29dff7e097a83a68fbee99496be736cdb56529393fb87d47170e092a3bb5e04b",
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
    "sha256": "84c3083c48f00baaac9cf0ee5faac8cc206bed702a07a4eaa8274a30b2f7422a",
    "sha256" + debug_suffix: "f87a8dc5f01d0ae66163535bc382edd9e1e360ec830d010347eca0402f3315e5",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "de5e605e5bcb4548b0544de5cb0c47f9cf4cc67eb9a1629d5caa488509544210",
    "sha256" + debug_suffix: "d3630deaf896584b398fcf3fdd48de0d3f4d9b62e04706c8390bef704b90f11b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "3ab4ffb5da3fc5d42c6a52a0f5cf615c04264d902abc2bf06e73b663273c8e24",
    "sha256" + debug_suffix: "e60352288c549c829a3c53c216dbb3ed63cf9f00ce08efb4125fac4ce8970d44",
  ],
  "kernels_optimized": [
    "sha256": "18689ab12320ff3227196fb89b4878998310d8462e8f7e36e937355e16f759cb",
    "sha256" + debug_suffix: "b19cab652b598ffc378bd8f34aa4b17d287694f678570b083dd905e21eb5703a",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "21a806abd775a5e94ee1aba77c55f423b6c7f28e6ee719638dd8ada4095fdbbb",
    "sha256" + debug_suffix: "1dfd00395eefe367674ccd605acfc6a0bef12ca34897f3438a59be71e3134f16",
  ],
  "kernels_torchao": [
    "sha256": "c42491575384a87bbde1827f5b6de79d23f717e0233ab6a77231b20707c455e3",
    "sha256" + debug_suffix: "96b77233e51f13f616c8d6ee18a298d44282d1866458ef650fbc8a89e7fe3533",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "158209c49cb2e19932719d06e6a696e4ba3c3ae3d4b778c28b78e5304f8f3a23",
    "sha256" + debug_suffix: "36351a4006494789d377f90b9b459d5a0286850b9b9a43f69676b7b5ee20f19f",
  ],
  "threadpool": [
    "sha256": "878d64b78967de86242c36188fd446484d17cc18c3ee98ad060b7452688b61cb",
    "sha256" + debug_suffix: "b6770e347a40cf4bf8101565d2dc15eb51a9cc7f1275675fa198dae0b9578a72",
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
