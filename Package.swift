// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260907"
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
    "sha256": "80abef7575abc65cc88423548587774fc3f4c69202dc50c3df789a1aceecebb6",
    "sha256" + debug_suffix: "8f15500709b64bf860ea7c025ba3d31cc5a2d168ac85b68c69af0dda4d331411",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "e0e9433961e9e67b709e66bd6defb17e123a20fc2fa92645d35872c993f973e8",
    "sha256" + debug_suffix: "3e961fb60872b4ff0e452a82234665be0aefe542cad460dc8e2d5a426d13dfe6",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "28215b15542326ff2b7bcce828b3f5f7301b7197016aa89708e8c28ed2973913",
    "sha256" + debug_suffix: "3c6102045001f1f65331ccc1ed1541531e16d07069e15a2eda110a04ed37c6af",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "49c878230bd8eb124e323a148c08f44cf68aa14b1ec7ee0b9c37a667ead825fe",
    "sha256" + debug_suffix: "ea6ef04aca328c9fd258858be9c980a34e9d8467f2bf50085ab1d793b5651610",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_dump": [
    "sha256": "c3f868b7237f60638119879dc5cad7b39d0ed23b5f54873ab366930c2a048575",
    "sha256" + debug_suffix: "9227cee33cb17599e1b188cea2955631f8106e4efacd70dfab7e0ce9e45d7ae4",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "b620cd001f47b8a1b15381c0af910d9a51918849843e1480a0945b51fe03ef0a",
    "sha256" + debug_suffix: "08c7396a8d06819f4197de3d081d6f66cdccb4e4a97ca3a8a8da0edd33c61060",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e9e8430db4302d60af121162b1a26215597a05e1302aa475e64563762c68e080",
    "sha256" + debug_suffix: "f64f05294a3297c1c664e5b589a5c0eb2357989e1ea14f56ed39000bc768be44",
  ],
  "kernels_optimized": [
    "sha256": "48683d97c318cf751648bbbba9343e0552fbecc89e7107f7e95fd5b1ae98f03f",
    "sha256" + debug_suffix: "590b2c0ee8fd520250d5b5a6fcc0b89361982f8e57842d427c82e1b2a3d117a9",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "5605555c2e176e85a6322f3d9d87797caa80a0cbe22dc3559491f5aa2db7c2f1",
    "sha256" + debug_suffix: "629106da558c6a3e759793b4ae73fb86dec30667fefabd1407bc27e771de6443",
  ],
  "kernels_torchao": [
    "sha256": "add4c7449071096ee6a6423c4c63202c7342fe85e97805608e0031ec0475435d",
    "sha256" + debug_suffix: "4c7cde842fa31ad0ed44d57518be7c31b0bc39ff69858ae5b65fda7c948b5705",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "fd7a4bb561d1ef44819a9f119d7fea989f34e4790d7f32238e360c840ae41801",
    "sha256" + debug_suffix: "e8e4eed614b8212837b1c320b278fa52c6705bac4a99b0a29e038fe539fd6ff3",
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
