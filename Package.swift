// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20260930"
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
    "sha256": "a2bf71809e124b31834026f320e97ea1ffcdf983fcfd5eabdf3c5e28239f01ab",
    "sha256" + debug_suffix: "9c362b272340df13ca12cdcdd0054232a4f3401fff4615d510529f1186a3620c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "f4fc2f272c3169d5dadf7821d60c29976660d3a9156e55811a75be7255fb0c42",
    "sha256" + debug_suffix: "5636f3ceec9b25dc4b695b8617915d1efcc2189eceb7937c22c673421c35dbc3",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f67f2340e198ef5a020cb1cd9f07f939ccc7d6382f4d60656db26dca1c25be3a",
    "sha256" + debug_suffix: "84e2abc4ae7fe53b7a107f87446788d3250e4bfd0a986d973a98c9c263527e5e",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "600362e6abf89865d19e31ff958c88bbdd44601c1a0b92f4d08ec6dfc6cfe9c5",
    "sha256" + debug_suffix: "01b33162dd31b7853cc4ddf21f5a19705b7b0ba56972a2e83e2e90a3c165aa31",
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
    "sha256": "d19bbdf1bca85cac91fa5b2b4efc8e3007f0e2eefc145457eaad5c0e683a05e6",
    "sha256" + debug_suffix: "da51d3c1e58fe8212bae490bd66ff567c4a9cec293369664cc8ebbd6c5d43bcc",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "56c2a9956018d7ed549f1a3ba1d7b3aaf720c5ba88a2a8e434b0d432ebc2159f",
    "sha256" + debug_suffix: "62064c4ca94339072ab3ae5bbf427da40277d59e89104e2d519c267f48f5b33e",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "42ec073b1befdc1e6092a47fbaadc74db1c89880f1a191339b09db6f91385dbe",
    "sha256" + debug_suffix: "e6b24203842bab4e1d2ffcd03d06454061d1f5b6e587e93baa705395fa7ee977",
  ],
  "kernels_optimized": [
    "sha256": "b299483c5c2238dc5e72f8e514812accffcc6a64b9a737381d429f643e0a7869",
    "sha256" + debug_suffix: "c52b40e25de518596ca66279d0defb939954a18e33dea2c10c07b44180d667db",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "adeac85a7be76d6520f27940074a3946bc866c52da47b5009711412bd83697d8",
    "sha256" + debug_suffix: "af7183d9ec88c80d77eff2a2df6f5623c6ca684b178de90d26356ad89f3022ef",
  ],
  "kernels_torchao": [
    "sha256": "24053f63139840ed83d3972b03574dc18bc1a740e37b18a78c8cc72b800c6340",
    "sha256" + debug_suffix: "0744ef960083456f92a6dab501acb7427e198bc7ab44bbadd9beb5488995e59a",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "ed1fa9506384e94c68366323df8533331a8ad225ee820ce4e8c4694e3602fc11",
    "sha256" + debug_suffix: "57a9678cfee8110076ba095fe93119b77fc3cc3df5bc8efbbeab5dcb992f21b0",
  ],
  "threadpool": [
    "sha256": "0446f4c1ba647051bab73c743070a640e106faa11ca034e700c6a07687485027",
    "sha256" + debug_suffix: "493db987e0e1503bf3f58c89f72b64861bb8621115caaf3a4fc77aaacc618282",
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
