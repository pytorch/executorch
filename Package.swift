// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261122"
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
    "sha256": "ccba0bb771e7e45fd8a0b8a18ffdba446bda527e59f35b2180bd7c404910ab3c",
    "sha256" + debug_suffix: "a2d41e19fc6fe51fa79c7c8a6e63d29884c26eef080b95923311167a30cf2683",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "319471ff6f7dea66db9e59cd37711c83bb4eacaf9f700b5e94252e5f04b3a747",
    "sha256" + debug_suffix: "9856519485c93d4c5667d2a7f6b83d0f81a7817f8d2f103eb22c1f353fcacdce",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3bcd0b40e1e4ec4f63f2c9f7f0ff2ade29b3c4184610cb1de1085b08adc25c66",
    "sha256" + debug_suffix: "b8f0baebbfb773826267e7f25a42503adb8cf540b7f89c82fc947c176e02668a",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9e6e83b2ab457bf684bcc6b2f433d3834547793ba3f369c3f297ceb22cfaded9",
    "sha256" + debug_suffix: "6197dd7ea201fdad831644b8bc22a9a9dd00c58902cb31cb2c79f953e430f24f",
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
    "sha256": "3a6c7b552ff4e91b1c3ff2c7a4c9b4c820c18e9f30a3f8c85d18525af75dd0b7",
    "sha256" + debug_suffix: "398f6cc06c1616fcfd17cf96b8751f99a09d5179d1d7212024d55cf6cad9dc94",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "8fc9c3a9846912d8017770dfbc0e6b682d10c082cc98785265b235153b5cdea9",
    "sha256" + debug_suffix: "1b03cae0a007d7334bb5e8068497e3f0338df9607dfcca27fa1ddfaf1acb1bb7",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "1daac453954cb53186ef91010fa18e9b92a9b3fb448e52c1f274863e70e1b3ae",
    "sha256" + debug_suffix: "9ed4f75b6d36a1fe655b553f3a2484e308a9044a2a5e66b9b10b753071624d2e",
  ],
  "kernels_optimized": [
    "sha256": "413682b01d9f59e0d89e2b65e7959e2fab9b79a61aa1c34095ff6d1bdef67726",
    "sha256" + debug_suffix: "31adfbc4f2ebef8bd24cf9acd2a44ac2f0944dc2f01da887e421fe967eb3724c",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d7cde0fe607634401b830928c9a61822ba79ed664cc4a1d107a42be821ddbbee",
    "sha256" + debug_suffix: "5fe197b61cfc9b2ba4f531d14eb1bed7e921f359426737d5f1a70362cd287d9d",
  ],
  "kernels_torchao": [
    "sha256": "f25808fa34dc51b6a9262e69c5b3b1da110791cc236371353c0d57deffcad5f8",
    "sha256" + debug_suffix: "f1c5ce77b584753efcfa9e6dafeafc8d8a6f2df606096faf9009fe87e2e6a1e4",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "bb4d507977bdb2787167fc79587dc4c77c54810d58f79a15933280369d8e9174",
    "sha256" + debug_suffix: "390520c67de1df10477803912dc4faec128ea281d37ddb8758d73fd1cea73066",
  ],
  "threadpool": [
    "sha256": "c9a23bbaf9cafd37addf48869f74e125596d9f504b4708b8d5a3768246277330",
    "sha256" + debug_suffix: "9164426555f2cb9fd6006ea072f563308e3ca68ce4b47e75d23433881d2f521d",
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
