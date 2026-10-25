// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261025"
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
    "sha256": "4f8584df76a15dac83323dfc8d4afcf77b0e78620819db5ade227deec7775b3d",
    "sha256" + debug_suffix: "bb9c9fc1d892d001ead17a549cc5e5f8931f92cb42eab3a5422ba8ad29cdecc0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "e34f85f57ee35cdbfc8d841c603a17eb30cc660ea5cddd27aebb6ab21428d766",
    "sha256" + debug_suffix: "85f15ea7da92f7936ae79b722fce0e347f10ed363b44db084c2ed548f3e031e5",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c21ec1f4c393e8d9b72eb2698225658f4b3e7f1284addad28128ce9394c6f8ea",
    "sha256" + debug_suffix: "4c5bf19ab591f95e7f645ed058d154d376c8914553c24bcbd0e6460ac2057e91",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "fc374cee7fb1c78537cec5248955f4a302c68d1a3b995d16b94fa3e39eae3e99",
    "sha256" + debug_suffix: "ee456b7ddd2870da119940cd48094c3d3c6b955d372a8e9100fb78d104cf9a80",
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
    "sha256": "ad88fd93e635dc8683cc8bf2c5553e8ed32403155261e19a217efd896aed5865",
    "sha256" + debug_suffix: "8ba9dab6258bec9e87070c31cad8df714c7f3a8f434a42a7f6c130e1f1650b25",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "30bc9620bf71d4ff633ef94c28c1a9ca5436c763b77aa739f40aca15c263c47f",
    "sha256" + debug_suffix: "c78f5ac5903e80cb313972efed68b419e410c8354e73363180202bce3cedafae",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "617638c4891ff512ebd21a988f3f6ab3d547de4af3dfd17319676ecb2b85877d",
    "sha256" + debug_suffix: "804abd3eb806c6b41927814ae35e4a16a4bfd6f6cd133653d54bc81b31c3a0ea",
  ],
  "kernels_optimized": [
    "sha256": "89df1a2639318c9ae4bbb5c5ac5dd0ea73b7eb2526293cc634bfbc1ea4e175e7",
    "sha256" + debug_suffix: "e3a8fb7ee0ff2c533285d21fbaf9cdce2e2c1a6fba0fea0c983e20b6dc2fe456",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ae9a0606b79cd7b0c726e76bd06a8deaf0b36b6b6faa1ddfb45bce611c8c3fc5",
    "sha256" + debug_suffix: "62713231437084fa6057df91ea9bce3c1c4d4079abc5c2a67be5d1a748164105",
  ],
  "kernels_torchao": [
    "sha256": "fd030513abee87dc3c1e9445d3c302911f00b72b8d81a61f0bf4beac0fac638f",
    "sha256" + debug_suffix: "1bfba26583ff9c1ea3416d6cfe82920beb7cba1be3f3364b6b310f1031ddb795",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "0df185c8ff09a490633ad73a7ac4c525090247fa2a65748d82f5db258a79c20d",
    "sha256" + debug_suffix: "7f116c5423685bcc43f838f611e6a53aaf130aa4acc3c37870810c477c218842",
  ],
  "threadpool": [
    "sha256": "cff16f8935e51c5783c3ea0dc9619e05e0c391c9eaa94bf74a54f15e186465a5",
    "sha256" + debug_suffix: "ce61066ce652a92cadb66580408ad258dbb15d4985d0082cec1fce03912618ab",
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
