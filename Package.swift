// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0"
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
    "sha256": "d194005164750414ce42cfaaad27750c8269bfecec03840a7176b2a2a3d1842d",
    "sha256" + debug_suffix: "ab289bf91bd840042e525d5ccc6dc233bd94415f6ed374b3d0e7b902f07a29b7",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "4cfed8f3ec80604d83f71debbd0e6630dee242ebea160475340685b9ca061b49",
    "sha256" + debug_suffix: "261544e0319846496f2b44d2e8491e369231c0f376f81aa13e18f29211905466",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "df8de2e5fd8134446a10c1e6ad2170d6d3f99658244ce8d6dab977307474009b",
    "sha256" + debug_suffix: "fb6a442249ecfdf6030d0681c1871da12368ad7f05fe7882db7cb364fa4963f6",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "3a861b6f6afd4264033a592309564932d96af758042b5c75574069fc0893313f",
    "sha256" + debug_suffix: "827d415aaae1bff9a349159da2bca5f74b912ad540087914c2cbb6b37d753d36",
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
    "sha256": "5656b534eec442be90e9eeeef23fba0065a2792e4c374f5aefa44a2234981389",
    "sha256" + debug_suffix: "147437cc77eff38be1977b504a6e0f8665dfaa79cfaed9f7ee1969ceae539e86",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "7175eee236b2822e43965791f30309d3bb839d08dab9ff03580a2a7521bd5e18",
    "sha256" + debug_suffix: "b3658e5fdcd895d6629f80150adfcb15f2e84b608997e677e1863bc68d77b133",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "debde1dcb8f36285c9ae4f1dc4d0d778eff2c1eec2d6805562aa397baa5cb81a",
    "sha256" + debug_suffix: "be9f122b6158ace97d9db513ce20f3c16f95997736bf1923aefb559303654a75",
  ],
  "kernels_optimized": [
    "sha256": "57016af23701a3d1a05883a9ed882e30e86c2365bafcc638be6d6754e9e943fe",
    "sha256" + debug_suffix: "1773e55afffcb9a8fc091fcbbff5fe307c8db837f42a772c70ace59dadff5147",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a4f5ad856cd3ff4bac7ccead48b7f681784990b178a11911be9d6d77a88ed82a",
    "sha256" + debug_suffix: "cea2734f4a7d6d4e1a41553e69445acf2c3fcbfd518986da0e4d5c4679c5b4f6",
  ],
  "kernels_torchao": [
    "sha256": "d2d43016d5d9803446b2d57eef0585c3978f421235ec7a91e81c5e998b1b8294",
    "sha256" + debug_suffix: "a9eaf221751d49cc21978d7be2b9f676dee633317a4ac5835d42ec4dea81c71c",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e036395eadd549dba201e5a9660976cabf978d54e8972d5c37d799d3859e5f4f",
    "sha256" + debug_suffix: "ce5b6bfb28e8dd95b8b7a11c8d87f17b73900707df369d33283d891fa0230824",
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
