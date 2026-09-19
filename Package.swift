// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20260919"
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
    "sha256": "cbfa1a98d48843944a0710b0eb36619a317fe750ccab9048720a70950df15796",
    "sha256" + debug_suffix: "5c8e01c39ab8db2bc42d7be4df0b1cc7862bde06c3e795d1a46fe13e063ef906",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "a3253ead82478865ba48c7cd25fd50354c3dc3db659876cce545d95c7b572569",
    "sha256" + debug_suffix: "91064efcba0e01c80d3d7f311b2e95b1e9835eda0590f226b826f5689cbd8f9b",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "090e811228d059b77e33d266ec6a52f59c8ed7bc38940281838ef8a30936f7f7",
    "sha256" + debug_suffix: "8eba56f297abb7bcc1b297dbe83b6ad25c0ff3f3751f8e548c45cf0e55f882ef",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "2477ff80b6b85162ff3ede35eb4111efed91c421f7de28dfc7e328313a27f339",
    "sha256" + debug_suffix: "82aa7c4d94c465f15a53eef9904f647daad50185c7d849485586419c76eaf73a",
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
    "sha256": "dd56f23b8d19ed861a893fa802e7df6c8a177b808e0edf0e32bff5051263c891",
    "sha256" + debug_suffix: "fea1b4f978db32dc8619ce411dc277e645f71e2161aba6fa012dbfe9de0fc686",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "9d81ddba3fae3ed4b0457e3ed06d4848ac8cd98690c5d4e1af1ae0911a664c5b",
    "sha256" + debug_suffix: "7915a3be974e8d22f29df891c6a5862da067804b68bbf7aa3fc629ef1aedd748",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "3f74f0ad99cab834eb9cf488bf9cb10e7287747f529f29448e7340795edb220f",
    "sha256" + debug_suffix: "6523d836656ab72929c499c3bf2087e0f6c540bd8e3c3e918acb5615bf71dbde",
  ],
  "kernels_optimized": [
    "sha256": "2e5980bea2d29bbb9e1dae2cfd955a3d46c8ba876fe49c73d7be0bcfd595a79a",
    "sha256" + debug_suffix: "57f7d6496b23b3113df9d1479922df8d6f03e5156e88b7d5df52ea2c536e2fc7",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "51107fcba03d25aff5eb875d49ed344bd1305c4ce984ddf8a92c056520c568ee",
    "sha256" + debug_suffix: "3917fe512a1b839e34f6b882bbf253be12bfc5336cde2d997172d1ae380930f2",
  ],
  "kernels_torchao": [
    "sha256": "44c7fb326397273c255bf7ae3b40127c2eff36723739c739449ad157d103c266",
    "sha256" + debug_suffix: "ea47606cc748ef5cca97e39d23fb00b91baeca625e08e7b678b2c71ac9e5a8ec",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "7f129011425cb5d6b531032ffad3a566748a4d537d7477f5b156eacaa5227c1c",
    "sha256" + debug_suffix: "274d4333a683847e69c2b801a0e702720d7dead7bde794a832d28894aeaf17b0",
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
