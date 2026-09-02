// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260902"
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
    "sha256": "d7548d8cfbef50d8c6763763eaa7217e8a740076191accfa133e4358dbe855ba",
    "sha256" + debug_suffix: "9fa4a853ed10eb3b3137c922ed0d8787a0d9013c80c816cfa7bfd39a450e70e8",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "7eaea38504d488f50d864b4fcb3fe9ba36d77de5e8abc61627dbfd9790540a9a",
    "sha256" + debug_suffix: "d5d6ed4165719c75990178d761020435ff6cecadbc6c76e12f840bead14f78b0",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "21628076be4410efd2ceb7af1d63449ad018c2a26b89af8fc744bc1ab41af38a",
    "sha256" + debug_suffix: "aaa062d4f5912b6e6441da741062a6863ba00cc58f0497820b6a1d209e07fd57",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "cc106b8095dcc18df6e11eefb36798f83cc6e1cf9db1253831007df548d088de",
    "sha256" + debug_suffix: "3545566fc367fcdbdabb61af81c6f7b2233abb931b0a4097ea3a736d5e76c6c7",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_dump": [
    "sha256": "86a147565bacb38f8d3b0b573537fe7b8dcd98063f25d7b89040a83cab4a403f",
    "sha256" + debug_suffix: "e9efc2f0d2a4e21a5fb2d717b21d41fe9a2b9151627e29fe8ffd0eaa793a90e6",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "b035afa4b08bd615f00b82e914a7f32d1bb20ceecdfc60f69ed0a3c126e2577a",
    "sha256" + debug_suffix: "e39683d2e0016e8999644e9f88f630cc9aa36fb7a0a8bb2cea67696557c29709",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "649fffd224f03594ee8f1bb3a9dc8952bff2ffb37c97373a705bc4453a9531ba",
    "sha256" + debug_suffix: "1d370013b2e2d9f16449b156c0800e7cf5d368d868204abdff6be6a5f086a8c7",
  ],
  "kernels_optimized": [
    "sha256": "f3f1cdf9dc1fa8e048f43b993f115f9c7fe90e4f75bc78751ec11b0ab0ba8c2e",
    "sha256" + debug_suffix: "8b2a6ecceae360d2d5603d11dd8f50b7463eb4d87cbffd6d691b6f3492ede98c",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c76987b934ee78ad820f46b437eb6893ce90330af29aedd389c895f530ba3765",
    "sha256" + debug_suffix: "d8e25111d2f94646632e8b19e3a4c2d52543223053eef2b51b8cdb9112050f55",
  ],
  "kernels_torchao": [
    "sha256": "82dfc317420e0cbce75d0e48b48e3eb856d715ab84aefc3fe4347b90200a2294",
    "sha256" + debug_suffix: "cef1358b244be79261fe5d3eb517f1b8e93e6870569ccdc238152a762f14d28f",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "94c77f94fe6f8f507f5c12349c6fe29915ac27524e70b0b193e791d1cddcded2",
    "sha256" + debug_suffix: "ee98ea65d114833e658a7e75e716dec883a5c5cb0c58459c0b7b426b4fdf9a75",
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
