// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261003"
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
    "sha256": "78791fba101d8ddc05d6e5b469305ed9eb3a712a1475ccbd9a3ada13ad6ddd13",
    "sha256" + debug_suffix: "3ec9bb77708d2a6324d6181b62fff2bd631439fc6a07ccbc375539b8c761d27d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "91d490363b7a2961be37f11af8be64698cbb8cbd2735d66692f54e5634dcf1f3",
    "sha256" + debug_suffix: "8c9adc68728a2c6799c8c7dcea6e2c5939c0a06adc239fae1c35a231b7818343",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5c9d0d383b94c1f15872ab219da52d2a1066460a6d7d1ac9fd5d700942ce2820",
    "sha256" + debug_suffix: "637907d1b31c43397d9d7b088540336c7f2f44d14778b53420c98b9030038780",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e79fd6ecd8fc56b6a407a00f5662af11fb8b9cb40153f1e80492534575c01975",
    "sha256" + debug_suffix: "bc57dc33924d6f99e16d989c58f88502bc95448c7bc0d5c151fbf3d3fbf1d101",
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
    "sha256": "8c37d93bbeffbd33598d2580657b8c4e9425205b69b23fe4a16def1c86ff2250",
    "sha256" + debug_suffix: "eba94736594fbae5260119eba89aa3c78e810caf67aeefaccd8f7f30d38b3140",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "77c25a83217aedb16e84ba11d23171f744988f10755a7a827aac46774bd35b48",
    "sha256" + debug_suffix: "38f4925da063df344a005235079604d39324494d70d263d0a74bc15dd58d4e4b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "4cdfbcbf791565933619d399cb3730ebae24fc41db875f7938861496b195ee0c",
    "sha256" + debug_suffix: "f926abb52b1c25637b840e41eeefee01b9be8974dae91355be5547d024742775",
  ],
  "kernels_optimized": [
    "sha256": "a063629540d1a8446b563bec4596d6f6f3653f9fc89782e85b992e2ae27dda58",
    "sha256" + debug_suffix: "be93bed3ffad0760f3904b36d8728ecd44de9ac0616ca41e50f1302149009d67",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "1342492e31e9479774698f8d2221ebba676c1337606e2efffe9232cae228c078",
    "sha256" + debug_suffix: "81ae0978fe705c357a153d897306da0c4309a6f144309fbbd927068844682c5f",
  ],
  "kernels_torchao": [
    "sha256": "74391d0ecdb5568c7d46af0befbe6bf992a57fb890ffb10459a7aa9da7ee9de6",
    "sha256" + debug_suffix: "12c19330a62268d6e41ae4f97a2a379a9c5c2fdd6d6afaba5d50b9ce36caed0b",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "c89496f4deb276d6ea62ca9f5924e494eddd4c2617f2bdbe2aab4ac41834dc14",
    "sha256" + debug_suffix: "61cd4f2e17900b18c85b8d35a6397d5fd39a92d862a14333d6d2b124e577032c",
  ],
  "threadpool": [
    "sha256": "7eb9fd28117df603f21c27efe5d533003bbdef7fef8107b13f692036a1c5009b",
    "sha256" + debug_suffix: "d19faa1672ceeca0aa815b60657fa4c26047ed4f01bb299f474a57f6475fae81",
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
