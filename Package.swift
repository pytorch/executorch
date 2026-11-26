// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261126"
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
    "sha256": "4244207836b071e9ce142677cb0652aa916e48820965796fe8676bfce14eb477",
    "sha256" + debug_suffix: "cc5ff30185e48a417ffc2522fea26578c54b2cfea2ccec4029ef0666ff268b0b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "72f89e91db3e87575187e9619deae913b81f912a1782b148dc620f4cbb04a756",
    "sha256" + debug_suffix: "66803d8449efd0ffdc6cf54c80b6e0c528e814e498c93c81cea1f3b1329ee008",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "93138ab3bb89d72b371159aaaafd256a6d90baa175753f5db8671ca8c27e947b",
    "sha256" + debug_suffix: "8e5d4f57a6c707022080ee7e4df81f6e0e87b79289ddbca0de13fe1d3a3a52df",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "332c6d62b4e0800351f82bef87ec740e831643b904acad0cd842c76bab7fd0fb",
    "sha256" + debug_suffix: "01a516343a8d73ee43569767b958ee212b4ff1889116853fd62c99b119dddc2d",
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
    "sha256": "28569db39868390847ae5eda3bd767b12826b41c54e58b0117177e8e1e1ec059",
    "sha256" + debug_suffix: "7943936376636e059e1ae9778365c2cded261140584a99dff7d5bcb060d4aa2f",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "f7b80ee11de7be68b2170a8ec02b61f3830e63512b5d9f7820bd2d6b03423143",
    "sha256" + debug_suffix: "d27967fd5d8cf711e2e267944e00a17ab10ec7f3e2a202302ddd70ca4035de5f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "48910edfa51afc74f22aff66b48bbee7d5821a023317c4521b504e1a69d150b6",
    "sha256" + debug_suffix: "ce3a4c8b4caee4458d11c5c3315d621082e39b8a2a5371b04d591220e56b5f3f",
  ],
  "kernels_optimized": [
    "sha256": "65ce941f8f061af638026fd426c3c7469b3b669d1af5c9fa6c70733fcddaa46a",
    "sha256" + debug_suffix: "5f2f6f82b7dd8d96c7dcea5e13d05038aec2752faf4106c2fea1d06668e2426e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "cf5b30c2cb6ea2eb95f9c51cc2bbd6a31f3e95e61189676b13386164d2d44ee7",
    "sha256" + debug_suffix: "9ca9efd0e29fffb052a8f3221247e5062987468d860590f9c1620a2debdb58e7",
  ],
  "kernels_torchao": [
    "sha256": "b38376259d4c52272d5d36f48e294036c0a4478eaf1be3e6649e774b1d94c75a",
    "sha256" + debug_suffix: "28bd79cb3052f05c90c0791932880f416ecea455924b23115bfe9dae8518e9b4",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "5d777c225d88813de5b3bb666829d49012c73f3f40f38d8829967a11b689e1da",
    "sha256" + debug_suffix: "24368c71593dfd4f913c56cbbadde76add763dde073d0a72cb5274df5672900e",
  ],
  "threadpool": [
    "sha256": "3e1f3760da09b4865de598290fc352877d9512e100445c27ef22204aeead9ac7",
    "sha256" + debug_suffix: "ab2c4dfb803052f3ae2f3e97123d9d549a4d25c1c1d4d11df532a3b0d84ac1fe",
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
