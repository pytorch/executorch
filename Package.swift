// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261010"
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
    "sha256": "755a21475f132e8beb6136e081d2f62e5b31f5c7eea566c69a4c355a9dc3e336",
    "sha256" + debug_suffix: "daf8ffb5f95be30d2074196483b844f960854180ff7085c6f6dfb0fd552ad2b4",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "10c019f1a4f700a221352b47f391f2b0d887964c3cc63ff1d6c39fc879a595ae",
    "sha256" + debug_suffix: "23973f2db7293cc1940d012c1bec93c82380cbe18726546883e915a3a97f7509",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0726a770a12bcb47f40db2edc0746685d1f80fa19d074bf38d86269724d41eff",
    "sha256" + debug_suffix: "a44940856eb7cf1482e8811b9ebbdde1bb1adb4580cf506b799d058eb5ce7923",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "4085f64a3bb817bf0f07f652954f0a724b6119ead7662c4740b2b3b55637cf9a",
    "sha256" + debug_suffix: "80c0af0da7d509ad2f3af0f48be3e94d5080a881dbe9e07246c4900624d2461c",
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
    "sha256": "43d9bd88cfcaa32f7051f35cbc0af8d6592a85de2daf9438127c8352dc34408a",
    "sha256" + debug_suffix: "485747f0c50d0a55f194ec2491fed9f8aac49c21961c897f57a193e5ea4b4714",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "da725a4b4fde202eab1627a7b4835286b545306cc497eb675a502fa7aa36528a",
    "sha256" + debug_suffix: "32b8306c9f5788a10c6b2d2a7cbff4443194842fd01f9d3f422626c806270456",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "3f1c06d3d1ca20fbd4d9dd42742d8b8f681c0f23f3d5b016608e6e89aa499a66",
    "sha256" + debug_suffix: "be3a6fb9395ec9f36aa2e6594cfe9b729b2f10d8e7b19cc94108c3ea7b281589",
  ],
  "kernels_optimized": [
    "sha256": "5c4adf0e78cd3f0d17cab8b5c2e2159af8ee4577c9a91893e68d9636ae1bd9a3",
    "sha256" + debug_suffix: "5894d2b275f07ebb89cf762122dda234d9172a7d0aabf5e0bdc72d31ba486caf",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "640f3ad9e91e17dba8832a626c56223aa019a4c4468783982e1bd80a86875925",
    "sha256" + debug_suffix: "72df2bf8f67aefa19fcecec092c9a7bf7d1283f959e917e4224d7403bf5a7cbe",
  ],
  "kernels_torchao": [
    "sha256": "e4bed37ebf553169434b774713867e960fa575ebd4d65f853a02b63e66f0fb69",
    "sha256" + debug_suffix: "ef3fb001a1387a195f02143c73023920be3b363ec66eb9c7683dee3da3763199",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "4a626143be9269864bd6330e70bd94db7646623c5b3f3866fb9d55fa04bce49f",
    "sha256" + debug_suffix: "dcd5131e4b7ea0e28c63f4b9d8fa4dc0a9a9f839fcb24617a5bad42026a45c3b",
  ],
  "threadpool": [
    "sha256": "bc6e5f13ac1b540a4d0c69cec31966ef7db7ea359543fa4320d43df735e2b7c2",
    "sha256" + debug_suffix: "96b1a58079fc56563c75fe1115e536a050b8f1db81457bb08f97a22419c9e1fb",
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
