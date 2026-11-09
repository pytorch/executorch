// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261109"
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
    "sha256": "47b71988f5411af9712eea7dbb3d827cadc87ee431e63aa27b8accbb4b1016f1",
    "sha256" + debug_suffix: "87f293f29a70c04d417bf0bdeb0ddc9d484d3a81e0f89f3d811acf1b8230dac8",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "83567af86123ad2c12be8298bf30dc776f8a06d43e82743959988180f9648063",
    "sha256" + debug_suffix: "ec3d716a0f955689fd16757848c6e72435caeed3f6b07e1b82af62330d312969",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1bfcd43e1182d1525b4b42be1c1900cbbe3e08225b72ab71310ae3adef542a65",
    "sha256" + debug_suffix: "f7c6b83418fae2c13582c8a037fc233e58ffed9655d1bcb1af07c01c7e04a6b5",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "1fdd5cb51cd08be2abc3593b88b5a892a89f3bc93cff954597998cf4d92ab6d0",
    "sha256" + debug_suffix: "6dfb93003ae0978b6a2c1eb6474c6d4c1dbe9ccb3d6636e269bcc4e30b6c5337",
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
    "sha256": "f3cc264a6891d9e66d716fe791d0518010825f214701c277d475f70d292c10d6",
    "sha256" + debug_suffix: "605ca80040b59ea07a7db6b2751dd69af0d48514bc26d1ac380e50d772b798b3",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "71c3389ee2b6a89521bc42960572a24c4dadc519a7a01aeedba1546a7657fa63",
    "sha256" + debug_suffix: "47ba95fe42dee934d243f0dbe335fb2007b065584db444dca94a88dbd6938f0d",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8bbf2d09135b90bd34da22b5b06b4a2b0d89574cf92d4e9e3a09f82031da1657",
    "sha256" + debug_suffix: "dcca5b0650325d3fc2033e9cb7ca2cfc3bcf01f54d0dfa6973f668c2e6da0504",
  ],
  "kernels_optimized": [
    "sha256": "abb4955e385ca87f01697b70bf91767191e338475366207875b75160ddf00f0d",
    "sha256" + debug_suffix: "113574f25d370f24df18864104a14fd1dc0b5590f1fad669db665c83425e618a",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "fd51a0a73349af4a9c3c3cfb2e89005b43c5fe2e61155307845fcd1374d64650",
    "sha256" + debug_suffix: "557786ee33f240235a53a576e1c1863102b41e4e3cb0c83caca522eb859d0ac5",
  ],
  "kernels_torchao": [
    "sha256": "42cdc84d5931c93f5853b3b72581453d80af0d268ce2b0eeb2e7504c74c4f543",
    "sha256" + debug_suffix: "47c82e6d7c894de4434ad9efd379be0bcb59dd69278b763da3ff5c40c4600f3e",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "735d8aab77e558fc3412e50f5feed82e979fb5d682ccc5d66fed68d8c5ecd05f",
    "sha256" + debug_suffix: "fa61e372683f0a6134f9b3cad609343e8dd9bf184ad7689f9d8bab38043525e6",
  ],
  "threadpool": [
    "sha256": "a76cea439681636dd4977aff4901bded765f75539b837647c3ab0a1093f71aa2",
    "sha256" + debug_suffix: "10ddf33eb170455d9a5cdea2e8bd27e09d4aa44d66820cb12e8e1d08266e1e77",
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
