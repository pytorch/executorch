// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20260913"
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
    "sha256": "19fdd353c9436ee95bbdc0141179abf614dc94b4549fbbc50e17a2eaa856de71",
    "sha256" + debug_suffix: "d4dbf8a5dd6622f3612f9f60ed608e3b55b77b633acdc5f7faa76982b57a2e86",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "c7f8b3665892543bb02ca15d098cc4beb1eeef69c07294387e738c47bdf7ddb5",
    "sha256" + debug_suffix: "22734877a17b8511930fa96fc05c55ea14d92fe4a70a16af6b9263da35713bbd",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b78776310475a86dfb144af6cdd53c12d3523ffa34713595c78191f8ef8c99ee",
    "sha256" + debug_suffix: "861864db5b30f842be1d231ee86047e9d6cbbcf220f0e308cd0c2b6af272f8e1",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d7bb6fdfa3cfa51ba1ee7cc2a8c670b2dc96b710298ce422ec0e90eee3208f45",
    "sha256" + debug_suffix: "d9440d147fe6ccf4e07ff25f718b50f12576d6d8f4f7e71bba7cba486c8fa235",
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
    "sha256": "0de1d11d41de0ec24b4aa6fc80164ef47339a34c83ef3f9481ea79e2881906c7",
    "sha256" + debug_suffix: "b5c6ebd87e22f1d1c5457deefb636371db8d178198b4d66a29d3b209b81f9ea1",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "d074313deca53e10564600ae047201faba938dce58ab111a37177c0059570455",
    "sha256" + debug_suffix: "54b7a92bbf0676abb16670cab053e05b16d78138a803e8ddc61380edd51ae0e7",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "d6d3bba7a6509470d3aae5a505e124adeffefed20121f7808eb489cb0b8481cb",
    "sha256" + debug_suffix: "8bc46f7bba906fe7e32be46edbb8ecd02a64e289dd5b02851b4cb44b5fa1cc61",
  ],
  "kernels_optimized": [
    "sha256": "c24404f16ebebf55b3059692d3ec87da96e5782c3e3218f1cbc58ee317af9244",
    "sha256" + debug_suffix: "8fdf5f23081ba9b62ced9047afae9ac62861cad4287c81c01fb6b9060edb85c6",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "e2554fc1a61966d0613e62137e4ce04d7ca4f14a2f29aabe1527a3f99b8316e1",
    "sha256" + debug_suffix: "18fb0ddc86c75a22f4bc38aa10e74fc8e5a68644c4bf115ce2e429c78faa92c6",
  ],
  "kernels_torchao": [
    "sha256": "49f8e89cf6d077d163dab3603168600d38db232c107f2e9e971a114c9e7db8d8",
    "sha256" + debug_suffix: "5c4a422df3a1edf519ae3bc2e9a8c06fba861f852d9ca1d14b3ddea0ba1b8c19",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "57ced1f8c1d9819c35d17e024bbd0e84d3c132a4b6f4caa5192da56f4225d69c",
    "sha256" + debug_suffix: "2d9943c9d352fee2a7d245aafa047f766e3c09f365c9da0292385c21fe90b4ae",
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
