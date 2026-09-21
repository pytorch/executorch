// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20260921"
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
    "sha256": "caff1db5791cc3cb50ee5a31adaa39a8ed968d576341881e4ac20805144ac865",
    "sha256" + debug_suffix: "6b93961217c50a5964f3cf7e196f9579d70142e903783e543ea468bd92051e7d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "1e5a67e57ba5336a1ec32e4a22a089798ee35fb0c369ee699be4e05316d0e345",
    "sha256" + debug_suffix: "faf335a20732a9ddbffecda26b653bd0218cab45f7c93aed90ac908d9b48bf25",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b9d59e2bb768de962bf0d407400ef79ee25f885116a857553a1a7c60685ece51",
    "sha256" + debug_suffix: "1f0b75d8ade96e50f79ab89a04b4fb58c66316dc7ba262f604098b19b1ba4798",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "360d58edd8ef055007c95dff5cf07c41f7432cafe9b6cd66f8e8a91a9abfeaaa",
    "sha256" + debug_suffix: "ede198f360002200847f76f8d45107268425cdcbde136cc2d669bfbef7136cd9",
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
    "sha256": "cb2a1926a1875d859a2b660ee7bb4f9f121f1ff1f8dde6a45e25ebe180cb0ace",
    "sha256" + debug_suffix: "f34a4cdbb32bb9f30e7303877516efe176705262d6b0b82b9a308a9b1638719d",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "9e05de4cfadbbafdcaa610073f7a77adff7c482b9214f7b07a43d8cae8c8bd48",
    "sha256" + debug_suffix: "1339b8835826bd5eba4c0046606892dccc1de51e1a888b9e27a43ab9abf4f892",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "dfda1e619d9cef94bf7308e67d684343c315a23255d9d1221f03bb0a26df5b5e",
    "sha256" + debug_suffix: "d0e7ca52b883322ad641399fbc769488a529fb400f253f54b62d699687bf0610",
  ],
  "kernels_optimized": [
    "sha256": "1d00eae5237c7fa805655aed95250cfa199a3975d2d70894d9b6a5f45b30c43e",
    "sha256" + debug_suffix: "d8250b2b9db26ed64fdb59c06785ff0753b6f0394b886f3bc420a7187cefbcb7",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a2147618d2b212eaf55390e80c411374a02dc4654d4e31d09d0c5da6e63c5026",
    "sha256" + debug_suffix: "9fc41600ce0865019168cd32ac88b678241f700ecc3d55b15acd280874120bda",
  ],
  "kernels_torchao": [
    "sha256": "226def20a38b2319860ef7b2f26d5a2177668b467ce4df1a9f34560a356156ec",
    "sha256" + debug_suffix: "f7bd1835d4dbffaffc1c8ddff6b3e1ee1d0829b833402fb19f8ed6e82d4bf6c1",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "f275e9c593aaa1880c063cffe7fdf8b499dd9bc89d3b12b4bdb48897e82ace24",
    "sha256" + debug_suffix: "b066a5c1078fce3162b0c0f9ce3aed6d2e60e04ed8dd90ee6a88760dcd62f38b",
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
