// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260831"
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
    "sha256": "8c639024cc61f66ed5e95c01cd5bf1b29f465a71a143f396df8a3fe07b7124cc",
    "sha256" + debug_suffix: "5a0a9f4809a6cd252a7c0b99cf92ee4f61f73838aca065dbecce353bb2316ac0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "766000c5b82f1dbb119bf4118caab8751c55cba43d6f5cb34fd18ac41afa5424",
    "sha256" + debug_suffix: "1e4f0042e96ab3dfbe112c99f6475278a44d14ea15c6d7efcff53b98dee4a49c",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ec0ad2c06c28f6e94b429068d851c9643394c466bf5326b4a911857f6b50cfec",
    "sha256" + debug_suffix: "2444dd6eb244f3eee5c90fcfb1fe01eddcb214977fbd8dc6ffc783a5d078c61f",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0bc3cf9514fa485ba908edaf91059eb21b8a9043736992eb42ea8d6e4b96dd61",
    "sha256" + debug_suffix: "4ef6b291d8372749a8c5ec82f6a03ca761672ea82b7b4b47ac1684c99cec5396",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_dump": [
    "sha256": "8d77f77afb9750772e9846da1f5e4976c56e5b29caea0f1d0ce24f0dfb1a7e2f",
    "sha256" + debug_suffix: "b4c0dcfd063c173c23796888f24f220e54998baee640106023cd53cd23f95877",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "28bde425c1fba16d52b2f51e7c825c0bba872c0f235ee1b1a5c0e369be8ca7c0",
    "sha256" + debug_suffix: "5e7590bd8d93805f04679379aa4452d98b5a807ebbd98683c00a560f780b064d",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "bb9a49606b14eadb1ef1c602cc1fe10042008cb221de06b2147733993b4809d6",
    "sha256" + debug_suffix: "ff5dc5776dc950971be4ab0065060c39ba9a2f0e44b32fe14fee40a43a6d3995",
  ],
  "kernels_optimized": [
    "sha256": "cae60a13d28b7384e21d581de938ced3e9c8570381a21defda5350150dc3f686",
    "sha256" + debug_suffix: "9963a77ac30a59816098eb783188d4501020953c78ee37ddbe26a5da7ffe0965",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "65692d8b5d9842ffe66dbef3d73978a52dd5c30856aa2cb7a2af9927ba33ac88",
    "sha256" + debug_suffix: "8df1d6cb7d480fdcb9ebd41951f14a776ebe9be54c3f73b80df5b9503233f039",
  ],
  "kernels_torchao": [
    "sha256": "0e03f97890ee1bab6faa49ebe71bffbcdbc3a29f35ff5570cf650112628f547e",
    "sha256" + debug_suffix: "323353c6f8ee1c50c4946d554d7726ef96aa9bf17d0539f4fe4adb008411426b",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "4ccff27276e70848f127bd407494e6350c26f8c60374b87f81c0b7031f238a07",
    "sha256" + debug_suffix: "bf1b2676916a7e31ffc8a5549398083cd4081297eafbb5770b26706a0ddf2df4",
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
