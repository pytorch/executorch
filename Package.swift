// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261112"
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
    "sha256": "0e2dcece90b1d80efae0f5cfbf97ea49a0c0fb29af283669c7cfb537eeeb822f",
    "sha256" + debug_suffix: "5f7d8a0f8e05cbdc45e72245d05432c6de04104b6c01e6caa9ce1635a03ba14c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "e69a645aee0d36bb250c589a6d8a8f6fa415af3f1d730d62259658bd3d6b890b",
    "sha256" + debug_suffix: "c0e70725bb59088ece926c574800e5ce190d2f9371dafac67f4057ff2311dffe",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c5e2e4fb18308597e436613192622adfdbe12eadf6a989dfacfbd9723543b1ca",
    "sha256" + debug_suffix: "504a064190200d7f80d67ebe43d040db5be0102467a395b09f26fae40669d712",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "58034f31c63790c818f7f3d6ec5129a6ee5b1ec0991d752236f94cc6ccf73da4",
    "sha256" + debug_suffix: "4690e64bb35211df260c9ad2eb899988f6213db58a4ce1b97d63fd19e5fb0bbb",
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
    "sha256": "5fc8f62211ccf5e882e92e19d996b3d621bd030123a91f0efeae1fd03764ae06",
    "sha256" + debug_suffix: "a69fac79e27c83df51859138ab19900521942ea39f9d1369df19b8dca3c319fd",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "b1e89b31e98c062b68419497bd1a2cdbcd9724deeb8e26dfdc94e2d242ba0b2e",
    "sha256" + debug_suffix: "1906f48c626cca63c1f27c934676b853df5d50a6ed7db9f9e4c421b4770ab88c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e4ae4c8083a8fd854003c45656dd7691de8830fb495d9ebcaea3abc4db7390e1",
    "sha256" + debug_suffix: "422fc6842d6dcb7da0b3c727c64bb1c079c40cecb1fbae7937f12438f2d0dd8d",
  ],
  "kernels_optimized": [
    "sha256": "1bad6b72a99b128d4bc82870002191cf647269b1fb2981dc1d662d699dce7f97",
    "sha256" + debug_suffix: "f0440efdebbb1832355482422ed72fcf7afd8312241bc328d9be192172c0edfc",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "1012866e92fafb8e8ef0a9b942e52377299b37fff601b82d98f27a3216d21e23",
    "sha256" + debug_suffix: "47008868b041d5fd5dd2cba158247f0ce3df0b373982a794159863f5401de6cd",
  ],
  "kernels_torchao": [
    "sha256": "8c5ebdaa09a70e21666d2cb4aa12bc7b9d9efb0d4289c0830f7a5ecc25a4de43",
    "sha256" + debug_suffix: "8744851083931d6af25dfa265664336d40f539cbecb72ce310c1a4fc99cdffec",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "9e3908a34a44835c6e97df51797814e81f01cea13f38cdb1ef8218fe8bccbc9a",
    "sha256" + debug_suffix: "194e6119c9cd3151a02460a8366253014644bde2836c87674ff5ca3cc1b85523",
  ],
  "threadpool": [
    "sha256": "28b9be76b80db14ca52cbc3d6487d1c9afa92d6b732124df1759d13a9c93bc5a",
    "sha256" + debug_suffix: "fd7aea88c6a678ee3158791a821f1631dcf23832d2c5ffe741cf912dd2252291",
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
