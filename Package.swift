// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261107"
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
    "sha256": "2f8ea590e4f98172824c6c992b1ec2a9e9ce5dedd559b9dd19b74f656af33136",
    "sha256" + debug_suffix: "c5a25cccfa0f5b33b0ec7488800e67a9bceeec411dd0ab5d687a788e49613736",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "e5722676055ac07a9c3d908957e6d6d13f12512fcde2a9c80e6a35129c9bf423",
    "sha256" + debug_suffix: "10243d5ffe09df04775be4e160f7903fb4aedb786453535740fc1877570c7617",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "bd1c12b9fc01ab846aba4e17aca9c56a158ff58da6c5e1942a86b4fe27e09e41",
    "sha256" + debug_suffix: "4fec6920ebe01fec417107ffb11d8bb39b255e9dccb2ca30d0f03657018def6d",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f5b1338f05fc751ada1a62bf97ae93ba9e1200d8100f505ea991170261601e3f",
    "sha256" + debug_suffix: "6df774cfe82601e8b15d11d1f6edee807d8849508130f0fdcef342b1aedcb33b",
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
    "sha256": "a2aba621af8d2d5070667427cbb6aaaa86eb6d15ba3cd8b850256dcf2f616a74",
    "sha256" + debug_suffix: "3cfd4b2d7ffb20b8e63d2ed3653373111e1969959483faa5df1714c15710711a",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "c6f8860ec9a51a74e0e63ee1bbf11d273f19a95c4e7f64c9e81dab0c523530b2",
    "sha256" + debug_suffix: "583544f3ccead66823475261de186523e0eebf1eb99568a6644b0cb61c1f19d9",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "36b527f5fe709c678ee33fb38601d471d12544e6bfe2b4689e62040c8839d0d0",
    "sha256" + debug_suffix: "ab4109118769138bb2ca6f04f2e1bc19056667cdbe9afa8198aa759aee64e35d",
  ],
  "kernels_optimized": [
    "sha256": "7bc080e15cc67443cc8519907f6351f685b88a614c9e488bb3a78cdc537bbcf2",
    "sha256" + debug_suffix: "6823b68684628280edccee6bda1415a93b7bc3a9a1a247532c75320633682326",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "33bc7b7e614ee673ba268e0bd5086774de4120ec5f8140bcabd53911372830de",
    "sha256" + debug_suffix: "3671957aabf359eda2988668cd3b618a66c519afae25153315d3422ef455a698",
  ],
  "kernels_torchao": [
    "sha256": "b3cfd4a3122b38f990d9e7cc73fb8ce456712e9bff568f2fb61b82eebe9a9e95",
    "sha256" + debug_suffix: "8653e0967fc386a01ba04eaa5480455facc7ec8e1ed169c7579ad0ae1c3a620a",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "0ce96789de6ebc1fb953e0ff19f5edf67c1070da7688b8e5b9bc54dd165a10ba",
    "sha256" + debug_suffix: "b76aecb7ccbb136f9c6c0c78d756ceb6dde6a0b5fb0cbd47d664b6edc98b4050",
  ],
  "threadpool": [
    "sha256": "e82525d723c6067d7cbe536e733e333eaaf3cd78c325e2dd19d1e1f727628877",
    "sha256" + debug_suffix: "a5b4bbc20e0776d8a579a7f39131dcf19dca1484a9875ed8981b9d97f734ef54",
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
