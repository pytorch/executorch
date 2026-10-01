// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261001"
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
    "sha256": "d15e0f574e1f9c89ed71dd4232af5bdd4fa83071da48f149494da7c53e2428c2",
    "sha256" + debug_suffix: "6e2307f6b11769351d47093888468c5ada3e8e5c6811e6f9b4932a6b3b82025e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "0d1c08e34b0464f41a11ac5172ac71aed83a5ba43f1bccd43b4fa42427470e49",
    "sha256" + debug_suffix: "084325d5b5767ac270b1f74adce592e311307930ec600176d154d95d9a3337f7",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "467941ca8202df59da26e7071dfba745ad1e47884ec3a18ddaec2dbe77050d37",
    "sha256" + debug_suffix: "14d4c7fee695752daa1e05c417c95bbf52adc78a0a1e1629af2ef455cc4ddbda",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e0b7ffcc6d3d109ce71e8e969681020d9fcaf0218b683d4f5c7874a0a21ec2bb",
    "sha256" + debug_suffix: "71d385097be3ffa03fd6c478fad80de885a00b8404f9db8c98e0f1050a989260",
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
    "sha256": "b32fdfb8c28655e0102ea17382d49ac4397c6f4e6bc90e2ea9a68a53ab42d2a8",
    "sha256" + debug_suffix: "61f87735dcd22f80c92dc086458a8d577bb10ea651c7d25631c2e8253aadd8e5",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "8a9c510a3673dd5ec501ba1b823ad1c8c5a742133a5faf24f39073a75d9bc905",
    "sha256" + debug_suffix: "1df4eac258ef0895d26fe6493ce5a9d043348420083a14ee2e4b6ca13ad849cf",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "5392a08b64de5d5a3c0f1c9d9e88e7acbc84ecf65c4489e30fe3f386ae2e9d61",
    "sha256" + debug_suffix: "8e358d31a21fab3bc9c1b047fcba493ab3420f2328fe46b49955493e5b19c225",
  ],
  "kernels_optimized": [
    "sha256": "76f6d2a15bbbc7c88f9feb131d2f2f2cd412d51c61b956dae9a4502c70cce1a7",
    "sha256" + debug_suffix: "fefa8d349617537378feec13b786e33aea44292afdef85598f4aea04cc3d18a4",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "7dd9848ead88bc22293dfa3c2c6093af5adfb622cbd15482728c94a27195da71",
    "sha256" + debug_suffix: "bf8f89d75cb6f08fd5f140e25a23de4f209bd366ebf176a05d957e52e6f99834",
  ],
  "kernels_torchao": [
    "sha256": "2c900827f1f21773c2a7c2c501ceefd03ba8b6355813ac7285bf313d05c535d6",
    "sha256" + debug_suffix: "9ea48a9c0e245acb834412fdeca6e2bb202c2b802cbf3da80b540daa29722a4e",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "eaf2aadc4c9c1eecc35fc8fb64594f46075be6bc6dc821fd19ae78bfc65e9786",
    "sha256" + debug_suffix: "c02b3779b9ba142760a47af2b069e24488cc7aecc6f60c1a3db3ab166a4059e8",
  ],
  "threadpool": [
    "sha256": "e2f36625ea28a5b5dea6d8c489f35aa1fdef016429b395171dfe23e495241db3",
    "sha256" + debug_suffix: "280573a67472f3a8afe885b88737a3176d97c09ff16f460947411dd3bc9bfc9d",
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
