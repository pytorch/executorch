// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20260918"
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
    "sha256": "1b46516b2359bcac82cd1b0ce4f205f7b3fa51a19398f0a1d24f166718d5a8d6",
    "sha256" + debug_suffix: "b23ae3e048e29da9193c9f06fb27e13db7b43a1c528e0175cffbbae5788f4eb2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "917a7532723320d9d4c1ff9906650f4eb4e1b84ae77421f0dd8841f7678328b1",
    "sha256" + debug_suffix: "3d54615e1b97ff2a9c93f3bba288f0f8c5151b6024edb9dff868bb3e9b67d37f",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2d64db681b3536a50ce2bd8d0c99a7971de954fbbf5ca2d881d4a474338974c3",
    "sha256" + debug_suffix: "47029887c81633ed3fdd5a4ae2b39c393fe8a6a49cfa8efcec057168751801b2",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9c7a8136530533122dcc4722c4e3cad28c4562aa23d2c946c021182281883ad2",
    "sha256" + debug_suffix: "0707b71cadac2eb5fef9617644c85daa3f69585af1027d23aa2fcb0494172e40",
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
    "sha256": "093323bdaa476fc2f90a4da0e1a0927ef0be302c2ea12efac0fc23c0de5aa2ee",
    "sha256" + debug_suffix: "361b83cb526dc28d9813f46dbb626b80fb8688db954d0efedafc8ec68dcb39b5",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "02e3b527f70db165ade46832c17fa715712ce4c82730bde293ab8e18195e226d",
    "sha256" + debug_suffix: "0378c616925e688c5f4cf3f23d2fdc2b9bd51bb2e7ac604b4311c1e1a39f402e",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "374d38151d628554506a6e4017ac814ea8920c9cedb9144b49897f3e8ae8dca4",
    "sha256" + debug_suffix: "77fb0495b651562bfbe249f99930473846ef0186b36ef8c15278e401891bb6f9",
  ],
  "kernels_optimized": [
    "sha256": "10fefb648d7df00b7a2de650980a6a180cf376d16e39a840297054881930bec8",
    "sha256" + debug_suffix: "76e60f277270507d3a2fe40eac67d0de9272d247524d995c0eb7c6dff6150102",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ef485b36e645671ba2cdc3568c328cd52f6e81f26e30db985f2d4d542de3e10e",
    "sha256" + debug_suffix: "73544050403501bc74f5d092cb0296acc5ca1108010df581a4589546c3cbe85f",
  ],
  "kernels_torchao": [
    "sha256": "0a83b2d96453496c9bdbc664630b737164e7e16c4b5736c8def90b8cff4bc343",
    "sha256" + debug_suffix: "e1a9198d3b953dc229e17cc3955d2badce76fe7fa390826aba6353896dfd4783",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "44e9fa6ff2cc6f196d43f97191a3f75d2ee2ea47f06594b8e7e2175874f101b7",
    "sha256" + debug_suffix: "694798df60ceacad21aecf00747f31684428b6b5a5033841419da50d3b70979d",
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
