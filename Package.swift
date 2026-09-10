// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260910"
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
    "sha256": "00f3cc0e800bbef0d9879b8fd8c2f320849ed06a0f861282ed9e55e45ec80938",
    "sha256" + debug_suffix: "db61b2c6a164c5e0d63dcabfe2de5068b05f99e5fd9b3c77da602dbe2cdcf7ab",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "7bc565b0f9962890c65c359d7678b9506a468ac0652980ccbaff3549de48426f",
    "sha256" + debug_suffix: "63b18c1e1bee0785860a85d7bedf8018b399b19ca37f88998122a04dc7960f6c",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7f63de03246c517d55e44fd0bfd33e1284284ca6996c09735d070f3c93936e53",
    "sha256" + debug_suffix: "f6493b99ec3c5f519172a45a87a53ce34f92f9ebbae69156e71727cb13833d03",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d658f792b1fa6089f73b50e5999b07ac4c3c17dd6655b84cbf747b587507c815",
    "sha256" + debug_suffix: "2e2de6456471e7f4f54807f03e698dea24b78213c442c98c38103b055e52162f",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_dump": [
    "sha256": "3c30bb6ad4d94d536f5cc276ae73089de7988d75ec21885d196a7a379d48683d",
    "sha256" + debug_suffix: "8a435711a73f1aeff80ca4d3498f42f1c06176151b080f40a2e2ef7bdff0115b",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "c6b255a3e21f2dc89afad7058ac181283d6912b3220af904a04850f8433aa5a3",
    "sha256" + debug_suffix: "4b6cb6dfa2f36ee4633bfa6f60e790c3370f1da0483154aa509c1cc84042796a",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "ad924fa1c455521a8a0f12af693ea44e50c6ddef10e7869d7797f7dc19ea44f0",
    "sha256" + debug_suffix: "88612ae82f62332a465063e7d0fd1ec3e65ba1f0b8023329ef4cb3704b539c40",
  ],
  "kernels_optimized": [
    "sha256": "7abb52a172409c500108a8bb6f05acab2c2b17b834f9439ffd7835e5c29efce9",
    "sha256" + debug_suffix: "6f243cffc0cba15bc8b1215dcfdc24453bf79a0ff6d2cc3fb2df065c7ad9f10e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a3c2a39f5fc4505cfcf2b1d1026f3b8227888f9259a39c16eb958bcb8fe0ec40",
    "sha256" + debug_suffix: "bc701385d2a9e15fbdd4ee1dda883f16cde7130602d1adb8f38111cf9c9230cb",
  ],
  "kernels_torchao": [
    "sha256": "76b6fc50516def13882eeb75899d443910153f5b9ffed2468161d06a29020001",
    "sha256" + debug_suffix: "752a9a5554da67c417e00c0acc8b2b6a520bde090e0a982f06ee21a28ebd3644",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a8944245a5f9ae6cee6123b36f2c4bf07302a651445143d764c307dc4d12736e",
    "sha256" + debug_suffix: "8f79635d88632b0e2788ab2812f2d0c11cdaeb04e800a591e454c12ceff25118",
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
