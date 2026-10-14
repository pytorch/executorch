// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261014"
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
    "sha256": "f4fff9aaa091e3cdaabcd6db76b85d0e436a2a2c65b5f1715351615852ee9d52",
    "sha256" + debug_suffix: "407a423b26465383b77f4dfda8feaf0d87eb2ecd226264bf9f8fd12283c971d5",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "fb1a26256c6bfcbcd47355946c9a88eb5fa3effa9243f9dc48dbdc01e4e78273",
    "sha256" + debug_suffix: "62c2526103394851ad4658d2e8f40b0db68d469d2ae1822ab8c8e8077393aa83",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8644afb480e9e10de6caa9a4a4b3e70666ce84621f4664142d2d5784c0c6cd75",
    "sha256" + debug_suffix: "900661986e72eee30afa479dfb58be990970390a6b96afac1ba51c6844e98896",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "ea817c7ff0b7bf4d21e11f27d19f389ee289836858f9bf17225c744297a8a15d",
    "sha256" + debug_suffix: "5f84ca91ca4b360cff8d9b18b0f7d423f4c7918caab7e39390e80a4ecdcec222",
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
    "sha256": "35c44cf247b091ed918efe8b813e1b217f63f4ee3a42cba0cb0b6e85359284d7",
    "sha256" + debug_suffix: "9154f9dcdc38c174d88b35850f5b3c7bd5440a31d178bd64ca81b7e5632aa41e",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "0f11a1d1b57cb8d71ee27839db5b7e98999a3ed99ddd78c7de4919923b036164",
    "sha256" + debug_suffix: "6242cfa841fca3ab54ae1b98bcb904702baafdf0bc888ed9e616f366f18020b0",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "077ce9640dedf6b583220c670572cd7c050f17f7ecdd6560552322721cb6e3ac",
    "sha256" + debug_suffix: "8ba82e2e2d7a649de43e9151e1795efc41ef6752d3043ebca5234b2116e80036",
  ],
  "kernels_optimized": [
    "sha256": "4e4809375731de75aab22e17ded60f4086d2d10d193a5098c2256d330ffb54a7",
    "sha256" + debug_suffix: "05effd7db67755be8b6d2716f5ec177ebf0bc33ff3cca439e3ecfdcf73bc8f7e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "537722dc748a50aaacce6f1f2246f8c7be935d52762a1cd427f834ccebefbd80",
    "sha256" + debug_suffix: "26824bbadabc64e24381727084c243bbcaef22a43942d9bfdacf2351df34e679",
  ],
  "kernels_torchao": [
    "sha256": "9ddadb0341c83574a70e45f48f71d7e4399d1c516396192326be52277db7a3ad",
    "sha256" + debug_suffix: "1559e7183126ce6dae53da6bad1e016536f7d9a3ab4508238f2923f057c356dd",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "504e5d9ecff8a8c2c4e7b883510f22142faf4a585519f54233b9c462cc6e6f0c",
    "sha256" + debug_suffix: "ab9578b7779ad4731a36cfab43aa6fda4134adde2039d15a8c7d956b87dc9230",
  ],
  "threadpool": [
    "sha256": "7ce65d53b6e568e4320a199298bca0132c5be781ab4afc52e6f3841311202a67",
    "sha256" + debug_suffix: "9949744798f38589f4c791633f6945946fec005601d4c335ea16ea758d24ed92",
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
