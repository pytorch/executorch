// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0"
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
    "sha256": "c9714ddc6060f30f2980a287deb11dd44924d5e13cecb35c66bc2d588e39350b",
    "sha256" + debug_suffix: "968875b20ca7fba12557bdd0627fdbb99250c5e4a686bad5328d073acc41fa12",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "26dc30ae37b94feee95eb5e762103d0fff2742971e3970fd5cae88b64d0fd696",
    "sha256" + debug_suffix: "a6c594879a2d6c4b390471e6b0551b364d987b88bd08a9fe7f7b198f96ba703c",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "fab4e8af45e8954f4e08c13cf000a208ba33adae974d5b26e1e99f03abecc08d",
    "sha256" + debug_suffix: "860c01f8943c8e84d01b49b68ce84def93814b5b0dd5b9b8b9aba8e38bb28a8d",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "2145548f7210d2e2e67f9517536659b39f1a3451269df550c1c2326ce5ca679a",
    "sha256" + debug_suffix: "7ac6f8e0b8a7778a505ca30339abb68e2e09e74aaa0d15706bffa45ec6861054",
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
    "sha256": "44832d8f938cddad3af65f7a55551f1b36258d81d1ef8ed9737c9fba95bae28b",
    "sha256" + debug_suffix: "4d88fb80e73e6bfb34577ad8d457976b5c2961f730939bcf65528edd9fb48325",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "aee720a7a0d43e21bc80e644f798d4cd34e02dc6e76cfdfc4ad3a725f9be71c5",
    "sha256" + debug_suffix: "5f24e3f6ac8e4786a2965282ad4f50012f3f37222ad1ae53e5c5e2189cb6f34f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "bd1df6ac2193182a8276ae7bbde53932152da72c448a02303b5e527c7e2d3051",
    "sha256" + debug_suffix: "02a2a7bb946f4bbe8d012653744b6ef68527ec0a99b473b28305aa7dace459bd",
  ],
  "kernels_optimized": [
    "sha256": "0ef3114ffe9bc37f4e34d65cc9e55df36fe0be96d558a3aec6f2f0a7bcdecaf7",
    "sha256" + debug_suffix: "479609825893152a9b7b6716bb4cddaf082f42a554fabc08cbb5ff913e4aee31",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c1c7688c860d840a8549415ae2d14ce97354e25021f14cf23f3b5b6057fbcfd7",
    "sha256" + debug_suffix: "92ff9d7b89b5cc5d0172fc837290e21a3d43ab328a9f51cba5879b49fe5666d7",
  ],
  "kernels_torchao": [
    "sha256": "0cf5a253c894bed4d3a293252daf4a055ee4227d515ecef7f8611aed5c47173f",
    "sha256" + debug_suffix: "4ea6891dcda006bf74873bfc20a064b6d250e3ddb14b5bcf513a1ad3eef565b6",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "b99d9674770c4eb7183c971ed852838a4a9194c909bdb54b28272c4e402d25c3",
    "sha256" + debug_suffix: "bccc50626107ca5b11831970e6f09aa9e3985d812bc614dad9ba196a256f4898",
  ],
  "threadpool": [
    "sha256": "df8087643fe7dcf2ee9aca9e25157102642eb214173240d28a92d5416ab23edd",
    "sha256" + debug_suffix: "042f5c2cd34b7ae8841ef4e516014f943c3c0951e1672159da18c464054378d6",
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
