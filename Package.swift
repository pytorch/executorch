// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261119"
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
    "sha256": "5250e492937a442726631b6a529505cb1504c4d6d349249a9891d44cd5a926f5",
    "sha256" + debug_suffix: "a766abcb5fb2496368c5973248424acd929e495c45e5b74834c60e5429b3d349",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "a483695928456552a209a125c7e0e4e31d5d3ae60a3fadf643359a1c528a876b",
    "sha256" + debug_suffix: "f6d4d865c0713f012f6606d7c47cfa5ce15e7e5eea3dee1592f4d0be36ad7dd2",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a4bd67b8eae818793cbe52fbf6d3fd54964b69d67f18d209d8b6f643f74935d2",
    "sha256" + debug_suffix: "b9d4bddcd880856b6e6add47c28e936824cb33bf5a096674e9da3eadc4aa1ac4",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "4b957f01b9d7e26f512c962f70db972339b12f1b89fa0c2c9ad71643012b6842",
    "sha256" + debug_suffix: "b7c707417ab7cd1a86fa0df2084d39ec19e222f01aba80f243fb2a535ee6b611",
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
    "sha256": "539d163ba8ff8f97bd73ded92dbe3e21bd2387f8f2a3589336d238dfe495afe2",
    "sha256" + debug_suffix: "7831b3cf54021365c1d16ff3b5527dca2e94b68c96146e23d900e8e7a38ab96f",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "e70d42673907873a413f69e1265410ee60a341f4e60058a362569de915526fad",
    "sha256" + debug_suffix: "507d013f24a1a9d43185a01da9b3868952edc90b49513b84c62bf1c62fc8d597",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8de6e78cb3fed785f3ccd91dd933aae76d3e279d48ccbf2e3c548ce5ce351028",
    "sha256" + debug_suffix: "ba2ac80912c155e52d3aa5a08ac8ff06701103859b9326e05f47cff9750f57a1",
  ],
  "kernels_optimized": [
    "sha256": "3a7880a1550ece65be49557a15e1733d08ca7f047397abba2ee4a1bc65ecc3d5",
    "sha256" + debug_suffix: "0e4e907a4637682d8d0d478fe54e3a5164dd302b90f7bce0ee1f3d845962bd56",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "6716307dea3b41eed5d665f2d409c67924c8975e346f4737cc6b9724484f783e",
    "sha256" + debug_suffix: "27685f5ae4b1834e4095f1c6fe5f7f29dd452526dd548082506ec86ea4b50e1e",
  ],
  "kernels_torchao": [
    "sha256": "68d372162ce10abe0c9cd15708e7c2e369ada8b6fb286a7893ee105166b0e06f",
    "sha256" + debug_suffix: "6f8b609cdabb199c4454afa9e0b861dd859e9a07fb249eaddc2c8881b5e1c637",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "36a3a5066a4e896870c220901e1cd39679a299e1682c0ccc2166f5759430c026",
    "sha256" + debug_suffix: "438768503e92dabe3fc3d837001fc7436282cb8a71263c75d0680f99432e398a",
  ],
  "threadpool": [
    "sha256": "233211c227c249ae41f136ebc644542f5ac06565540a94fa7456976c139776b9",
    "sha256" + debug_suffix: "c9918aab11b5773e83fa7dcb650b9d16d192d9a7c86877036c117263407f46db",
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
