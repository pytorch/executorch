// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20260911"
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
    "sha256": "561a3254b757cf4ddf1eb0a94dcf5d5a475b370c8054de69d40653941c2b4c4b",
    "sha256" + debug_suffix: "c6f188e50b55b958e1d68961955b92e07089e98e1be9042914186a8852f54376",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "b286ea3c9f0b2496c53a578a1520444c06c8a233914e24a55de5581ec232eb28",
    "sha256" + debug_suffix: "08388de3f3cc1a9dced9d05133a3edda36df342efc4a48cf47b7fb4b10808b9b",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f42d68aa0a1e484b796a0ec1841e53c2c50102e3f38f09729e2c98c5a9a3ae67",
    "sha256" + debug_suffix: "a061db11410dda9cb9c2f0eb537a1df10b8671425c6d355fb699c186fd4bcd23",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "099d1136bdf764b447bc341ba240ccf0f1aab81de150f1cdcaba4142ced02a03",
    "sha256" + debug_suffix: "abe37f5bb0de415701b01b5996892526e166b454b1fc325f5d1c7b3c1397ec5c",
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
    "sha256": "568313fe99afce1de3626fb4989727fff22d12546042ae7b279ca8aef7be1c62",
    "sha256" + debug_suffix: "17e5cef86890fecdd91595c2b40deab43ab858833c96363e082a01d4a054fd2c",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "8d889d81e992525f50b6c58348d8c0d71b8997be5a6ab9ff4d83bb1eee361d88",
    "sha256" + debug_suffix: "9bad00a3cd4347356988422e1472d521f8f587bb2dbd5eedc32190b1f24d012f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "25e14e850448e65a826094fff78cd09c9736c0f5db5e27918d908a6967693c06",
    "sha256" + debug_suffix: "db13a837cfcb28d2d9188e95d3adc13e60026108c4f9ab93a72757ed7ef868a4",
  ],
  "kernels_optimized": [
    "sha256": "1c2164c334762b611bb78f767e581226bc2f7d637ece5cba776afc913351d0d2",
    "sha256" + debug_suffix: "fc9811c19d49d30a79e6d3dcd4ef406dbd8cc4734bbfcbc291ca50af16aa4ae6",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f72a7abacbd45e7591e5614da2b51485f425ec17024932181078fb27eb9073fc",
    "sha256" + debug_suffix: "c5b6fd6e95a8e53df8fc933ce2abde97ff76dfff4ee2259b9de0e3cef7951318",
  ],
  "kernels_torchao": [
    "sha256": "9e8b7af00ac28b1aecea5a17f2e8f88fdf7079ae9a0e890364356592aa42e151",
    "sha256" + debug_suffix: "78c330fa6531b28e735a3624ccd4113ea272fa393cb28954d464675d7ec1ea34",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "ccbad3604c20fa335091a13193b990beed3af550f31dbd9eda143ea63c123de1",
    "sha256" + debug_suffix: "7878b1321b53b05f9f4d1ab271b72b151920e72ed0fc221e6d1bc4c4ea187977",
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
