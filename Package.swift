// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261028"
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
    "sha256": "b85711749bd137001c0c1f6ab295dc64b86c43adc5bdcaf77b2ccf03529099b8",
    "sha256" + debug_suffix: "9fca3cbcd7494f772163b63334b943d573203474420a49b924f0ca4069b6ec84",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "8e1954014450c973fece2e2e3bb37064c368b300c0b9d10e8ab42345fc202e8a",
    "sha256" + debug_suffix: "249ab7bd9c20e95597979ce009ab5c9e0e6f0684bc5b2c4f80aa2c585f3fccca",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e4a8f94c1a2075a5f8d6e0afe246807c3595970d946ae27e769c8c43f3f64797",
    "sha256" + debug_suffix: "f94f540ce0aafe6b3619fca5f8bb0b2f0ff4f9c34c2049ab674bf03247bbb60a",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "ab12addccb68d3cc4cca939bb8da47bd21432a910d3cc087b2eb65201cfca847",
    "sha256" + debug_suffix: "51d7a2f855a28eb48833cfa8ac4b213873a81d76f95f7e70a29054becf8ede32",
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
    "sha256": "c670eb87354e10e1597407f8d09036a0c4b8dff921bfef9f7bb6404eaa1af723",
    "sha256" + debug_suffix: "e5af206220ddc3d421db602ce2b8ae6802f15ce63ba12f72757d70b166fa7594",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "11d6b589c965acf03d314769e03099a5b5ff3ab5e2888a809dc9f29684755fc3",
    "sha256" + debug_suffix: "55092201a6f274ca55fcfa29e4d23e7360a14d0db240c4ef5253eee0791e65bf",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "14094b499fc48b57aef785162d1a122e3709107a7af2819d0fd5161df5b1be50",
    "sha256" + debug_suffix: "4de8d40936049606bf9d46c99218870a75bd2952a258bccff2c1771d06f2d2ab",
  ],
  "kernels_optimized": [
    "sha256": "432cac0f8c5ae7b5ee047d6bcd7d165b4f658ae5b0b68a6121cfff3203365e49",
    "sha256" + debug_suffix: "670a4247beee0bc0ab3746138c51966a42fd6fdec1e033803c8f91bfdc2289da",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c7b746777909fbfa38038f73b382a661c755f225eac3d323c99b0a90c8be44b9",
    "sha256" + debug_suffix: "00b314ba724b76271711025a1d268c90cfaff14b8c92be030c8b22e108df5be9",
  ],
  "kernels_torchao": [
    "sha256": "b7e8d8e843aa287a39d22299dea9358c69e414953daa5e79d8f84848993cc08e",
    "sha256" + debug_suffix: "5601fe7a6244219b46083336dcdaf26cae529dd4375160d33b51984145bd3134",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "74b63f86da911a913c3b8f752f8d70a26fdbed117f319d809bb40cd071c672e2",
    "sha256" + debug_suffix: "73cbb1aa34cdf33f385a9761c00eecabf3611bfdb02c5368a1db30174afacfb7",
  ],
  "threadpool": [
    "sha256": "b9f16c160adef3507edc9f6d097a9806996eda7d6eb9a7982c9b5456c9451b1a",
    "sha256" + debug_suffix: "0320f12c686d33f9ac7096dc99656cd3091a83744963af707f29d99007c3c59b",
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
