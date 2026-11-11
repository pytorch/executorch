// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261111"
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
    "sha256": "6802a82b481fecf5bdfcb039dff5ebd23a73827e13541c1b3b7adec2638d48db",
    "sha256" + debug_suffix: "29750a01c58a95150a28feca236179e0709e5b8ebe21dd57035840513e6ef2a8",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "2c15da934c1894d84600aba5da3fea209dc83c8a133868387ad66b7105dbca39",
    "sha256" + debug_suffix: "e1ac37a509bf057de354288de0a51e76a4a6d64dbce55c9458056526505e0a3d",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f2b61710d9d643f8842d519ef123c7011655f61a4f5dfe13f8a10758977e0090",
    "sha256" + debug_suffix: "183d096711a7aef6aa079b42ce61fdb610e87cb16327701c9c1b16e771c92533",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5969ed908b4b88bc14c7e519151fea0a9989496853db606e540486f8f6b5c9a1",
    "sha256" + debug_suffix: "a857a0d032b29dc59791a9d5cdc04d7cdcdd5cd9b643ec426c45a69627538f1e",
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
    "sha256": "d53caea9f77337e836b2a6e6ca872a8e246b5f5821e275160903bab6cfea066c",
    "sha256" + debug_suffix: "43396f0129490e1adeb50b1016adc99286608eb1f9e6d7a456af6889b4ad3776",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "fdd3b1869f3f1ef1ab08ca92c52c807171f1a67d7923ae480290d1805d68b097",
    "sha256" + debug_suffix: "7d44379447e27481d2bbb02b3d9289c541edcd8ffbe61a30453ecce7278d407f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "b7115725005e53f29d1196785d0281367405fb7d700837c33e031a5d6e9092d2",
    "sha256" + debug_suffix: "07079827820d0899cedcbfd07451d575d558a31e8a040389f202552963c64386",
  ],
  "kernels_optimized": [
    "sha256": "adcf7399109176906daf6d81813b674d74339c6f0ba8f76f9b9159ed3a0cd89e",
    "sha256" + debug_suffix: "24a1a18b396d24f3ee20785a17ff9b0b8bb7ad8fc2fabc33a9b768acab344d7d",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c82b68e21905405fc78cca763d92a0af92dde62f20f2a6afe6b7e4a2a9ff642e",
    "sha256" + debug_suffix: "51904dd30132e713e8494e78b44b18324dfe69a124d61758e96369274977b6b1",
  ],
  "kernels_torchao": [
    "sha256": "250bae19ec076971496860cb73f5dd3a04d9f6ecde15c46229cc856dcef85bc3",
    "sha256" + debug_suffix: "9f9192f4c1abe98ded92435a94be56c84f27abfd00b5fb3a5db63b8a3b84c5cc",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "d75f9aa524143325481494730a66a8cb5930b7c9ec28bcd8e50ed7aa378deaa3",
    "sha256" + debug_suffix: "6d3efd99e25f1f054ebb1916dfa715164484066aab4860eab0f41e113f7832be",
  ],
  "threadpool": [
    "sha256": "0c90f4ee04c1f204b95018679745725651ce61aa87e7c1ba3964675313c64e55",
    "sha256" + debug_suffix: "bd8cc02f3ee36907368164988b2cacff83d4b0c90072eab94d02967832bdf2d4",
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
