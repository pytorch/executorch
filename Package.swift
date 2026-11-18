// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261118"
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
    "sha256": "999aaafa9e8750625932dfc432c064cd235a85d43739255f1540446be4153c2e",
    "sha256" + debug_suffix: "83b2e3a19b936f0c73880e0f708571da60f4c79f680a61849e1797a830ab7806",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "eb4a504d2269dade9e99d76e37144309c22e69ad030d5d1fae50a0c19db69d92",
    "sha256" + debug_suffix: "9dbef222ee44c134f0ffe43275269eeea315faf02012eea37c03f435ecc25c9f",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "34965e11e6825d993c88d20204fec973b41f68e3b2a1fbcda1a551e7eba87c94",
    "sha256" + debug_suffix: "97eca4db588b8c356a525152988b8b53d9009e001e4799319d6d88939e1035d2",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "8be3da6fb7d110292e2a8a68fdd3a0eadcdbe851817997054dd755c6dd3abdab",
    "sha256" + debug_suffix: "85d97326a2432c0b86001f409878fce5e9b7d2081be9d227fea2cce885093eca",
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
    "sha256": "d7d1063e34bf6445da56cb1e8df41b8a13b8f9450853f710f96a69e959376948",
    "sha256" + debug_suffix: "1f4ce9cf02ca2117e5c12220058184e2374535b175d92ec0c77b4f59c97d73be",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "17c2fa8487d57d3c13538bf5e6ebe9537ea73381e502fbeb7068aacd1bc230fe",
    "sha256" + debug_suffix: "4416748aedf2c6c1440af725f8e7ec1c50b2c27acf3d4f72768d5943ab477ab3",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "546dbf710d7312d9b68031b82914c3b2b210efd07d30c0a29763833ba1bd2909",
    "sha256" + debug_suffix: "91f4b8a495369e931995653aa4684620edb670e2c9379c1536c5b54d398e3d86",
  ],
  "kernels_optimized": [
    "sha256": "362eeabbbaef80d1fad9874ce945dc83691a4d1f7a188b1faf289c1ba3c61fd9",
    "sha256" + debug_suffix: "3e0c9d1ed16cd8f8cdd59f337400a692001ced2b186c5a83958b69a59d146091",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a9b8fc09112e5bc81c5bce9b09e94142b1c92aeb6722854005eec8857226d8dc",
    "sha256" + debug_suffix: "a5964650035e584025a545059c4a485490558f0f4fd939f7d63d571664b20123",
  ],
  "kernels_torchao": [
    "sha256": "be895ec7e54cfa900fb46b40e716cb805d24a5b7d27fe5dc55b8ef9dcbaa8d28",
    "sha256" + debug_suffix: "f70b94da9753f588c0e1ec09ce815c338e13e4eb6e08b3b48c10e39aafa2b550",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "81001cc043f7288a5beeee9c0d2278b183ac7850cf9836f4ecc00c773d4f4126",
    "sha256" + debug_suffix: "e250237b82da9d4b8ede23ca0d1ceb1c871882f027a513c7d459b2eaef86f329",
  ],
  "threadpool": [
    "sha256": "6f7a562099b2523a5e2dc750dc888e101174b45dd0a57ac50a05a1de3977597e",
    "sha256" + debug_suffix: "f3961119518345b39256c9e8c0d4f64dcceff95744f4137f4f4d36a7582610dd",
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
