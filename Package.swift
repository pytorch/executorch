// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260908"
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
    "sha256": "6760fd359c18fef42b103cab4ec5ccd0b0f85fccb7c78d1a1dd8a5d21e4efca5",
    "sha256" + debug_suffix: "2815555e9241effa93b246c32318585c75501babb1ba82976ae2dcfe95fcb3ad",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "d3034c80edf168c53924145f94ceff47311cdb41cedf4bad32454e6c6d8f47b2",
    "sha256" + debug_suffix: "68fdea835a8754310986edc5aaac8a0e8721858c1cf120604b768c1e0ceedbbb",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "dfef28aa2c2863886b8c8118c1a491e3de7cea535b6fc01ac4761381257b7259",
    "sha256" + debug_suffix: "2fa5914d8a97656071c8662333a2810e922bd6a0b40e0facf91d22bed3d71480",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "4b7a89eb66e3488e28f57f49e204bcb44093157e3d4c6b9e65540fbc6577edac",
    "sha256" + debug_suffix: "7d4385ecc320dd0326221a8dab09c7fe6da27f685402319334b98e1dd1122bb5",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_dump": [
    "sha256": "0fcf67a1fe9aafddd010c1e9631f5ee4024865a6c906608b21f73f5d61296ea1",
    "sha256" + debug_suffix: "eabcbafc87de70c8ba8801182475da3f1e2b2900c95f52451cc71db53408ac48",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "927613d9e20c5a228a26bc16d40e3c3ea87ca988cf1ca419f81eb8a3a7d51817",
    "sha256" + debug_suffix: "c6b01e78bb514481dbf9b5e733ed64c4a86edca5e9c692581941061d536b3773",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "614ab9066f906dc173115827ec5db663d9c06af3427de6c44edd50b7ba2660a5",
    "sha256" + debug_suffix: "58cee2bed25c2ea3eb675997313f8aab5cf8745d73596f87952e52b502ed6a86",
  ],
  "kernels_optimized": [
    "sha256": "e2822535ffe4389a05f0d4cf13c12027082201cbfed8ced9c20e11376b4677ff",
    "sha256" + debug_suffix: "3b3406ab54b97bf8abae5d62fd522c0d5fd43c2660fed6198220bc3b28925bce",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "6779435dc6d5917bc18b364bb7b766421a599de1d576efd8b7823a89a5e4e376",
    "sha256" + debug_suffix: "beee042a921f724a2c6084ba3aafad888a11639fc8413678194b69509f068d7e",
  ],
  "kernels_torchao": [
    "sha256": "dd42e85063aeeb6aeaffb159614075e080b6fc50c9333315fb95768df1496cd4",
    "sha256" + debug_suffix: "e6cf8a86a3c5b3cb36e165b17b8c3a9cdb33dedcf450b1a0760cfcbab48c8313",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "39d7954e6023624afdc0d2ade8817b952a19d61ae28016cdea5d6fbbb2820402",
    "sha256" + debug_suffix: "9742b6fa18332ea432f3ca2994344562228e3c3969b3e88c18010b3342086eb5",
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
