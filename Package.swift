// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261017"
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
    "sha256": "1a7b0c1fcdee833c19f9631ef6762744b3aa9182eb8a7b2b26995545e26441ca",
    "sha256" + debug_suffix: "d899f0f336b9cfd760779ed7ee25dee158fe76519a5c9dd65d1b0c33d56abb42",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "bd656d69fab2ac99edb06d0e1ad0f781c069a747e071ac933e8677dd12c89d2a",
    "sha256" + debug_suffix: "f7ba987be32ce2b87b3b2776228e4c41edab6592c61ae3a278eea5837d71bd75",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "816a88e2d9fe5cf98532a20a4f7836b4440b37452da321f653fcc4a0b7637659",
    "sha256" + debug_suffix: "ee3829a81ead66596c17dbd7afbdad55112e7fe1f3246549d6571d306157ac68",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "02d8676d459c6cc7eaa35eed44209c6066642a78444b8078e0825c9949c57ff3",
    "sha256" + debug_suffix: "1b672655ec3bc66f361685fbc135a8bd6e2cb4bb1d977269776f4062a6eebb9c",
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
    "sha256": "85b27ceb2fe2a22d1bb626102cc049d061dbd4d4eefac36ee41f68ed0c9254ec",
    "sha256" + debug_suffix: "ff481b0687c394efb40f32b94a5e194d7c8c601ab85c9a9dab8be87ff2a98551",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "2662d398073cf79d9a257936537fc566029376d7c242fef4c05da6deebefd26e",
    "sha256" + debug_suffix: "e6f59cbb22120c30b63d7f109029ec806ede6c85ae21472a7529c35b71e654d8",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c307c6a2d45791f7f538a5d3b1735f10d8884aeaaeac4e2247780c0531df0729",
    "sha256" + debug_suffix: "ae7354990b074ba67f3bef27f44d3a0474df676e2cd66822e82ddb81534ab35e",
  ],
  "kernels_optimized": [
    "sha256": "9280897afe423e9254c9c92698b00692b17945b626a8d8a218a96d0c95172991",
    "sha256" + debug_suffix: "9facade10ab59e27e8f33517ad966c0c94d32c2ad587e80640b365f6fb93bd57",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b47bb80b490974707a305ed15daf59244f593b85f87c972341558ac16ca385a4",
    "sha256" + debug_suffix: "a053b75f55b034e29cc4ccca7521a77e644e6901d07d95d3896d343b4e73f32a",
  ],
  "kernels_torchao": [
    "sha256": "24fdb0df21431059312886dd11ceb262a1a13ac54697c354492b9fbdf4e0c2ab",
    "sha256" + debug_suffix: "296153b9c3cdb5613c547a5ff080c7e99e28dcc675f4d36fbeb649fb85547ce3",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "fd1c4734710d839f37f533a43624a6946108e81a05baaef99cb959993cd05f39",
    "sha256" + debug_suffix: "426b02fb37d88e3662a38b13eaf5bd87620a4befcafe4be343c654793420273a",
  ],
  "threadpool": [
    "sha256": "d790881e2cf718a35cd3b61fdf2f937bac0542580edac071a765491c4c9fd50b",
    "sha256" + debug_suffix: "8501e90e17631c44d7d7d00809dc1c674718152ef2ac5cbefbe56c09c1c35cd4",
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
