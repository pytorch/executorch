// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261124"
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
    "sha256": "5d03ff7307aa3dfe2aceff7d1cab11f7f17006565ad5557a039beb4ad2825b0d",
    "sha256" + debug_suffix: "fb4076ca8ef42e933fe6064935874eec5fe9961e57da44b7f1ee0e003f672251",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "fc40213bc205937d215a1a337c4a993bfc776c7c595184c2ea62b2f9fb1431c6",
    "sha256" + debug_suffix: "e26094b2ac99cee38602bc140c9139f557181ea782b63f6fefbbeae298cce209",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0bbf984d5ecd49996d32ca5b790fc7ef187136d0dc4c249dee1032390d3e1b53",
    "sha256" + debug_suffix: "46f4c23d9867ceddffe3085fbb2055caf67c38facbd75707a9b9a4304ee50cd1",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "4f66146741627c9feba8a3d120a73943907a5e6b7337e2de7035fba0d701e05d",
    "sha256" + debug_suffix: "ef5f1bf812e57ac65cb139561039250cddf8e016f681815a10a604e668bc9a58",
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
    "sha256": "dd611bb9bfe1795d58d9437e497319960326c3c084124a492da56a39cec2b5bf",
    "sha256" + debug_suffix: "49fa06f1569c3829d1c342d2e7cd03d83d1bd59b8924bff3771cd63079bab891",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "2d74a7968e31723f50470c59cbc6bfde32cd9eff017dca58f3b611657c4ab52a",
    "sha256" + debug_suffix: "13b811e81fe1e4c9eed8599c40b10857dce3bfbe0da7428e30198bc842b429dc",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "dce3e3c77d8734aeecd0b83a4df887cf76f08859c2258b633ccbeff9f0443f78",
    "sha256" + debug_suffix: "262b4501fa7dc33255849d0eb3d24c33815df4f9ed8e437581d72b35c0c08cef",
  ],
  "kernels_optimized": [
    "sha256": "46805eacfb5c24f9a6604e7b634f8bae1ea92e4a8a54df1d64baa44f57615de0",
    "sha256" + debug_suffix: "cf510d009dcc0dee02d4f409384b5d7ac42d004ec0e7613b94e415e0bde02539",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "cc3b3faa615bcc7c94627faea04deced1cfd3510295725caf9bfb0ac753fb9f4",
    "sha256" + debug_suffix: "c1e4da8d99388686db7d9133f69057142533dc77deee79377cc48dc3ca13773b",
  ],
  "kernels_torchao": [
    "sha256": "491cde0361766593ec7f1bcd025c6f755655c90b4ead7c22bf43f8752e8eff40",
    "sha256" + debug_suffix: "69d3b95fb3e521b3fe25140a369b528679ac7fc3ebe3a55182133d265fc33f88",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "acbb52fa87d201a73e5f8a7659a244485017250b15e1d71424dde698d16b5ea8",
    "sha256" + debug_suffix: "aa3e28c30ab57d4105f140fde1541cb79b73a470cfe7a8456faca0ea71ed5ea1",
  ],
  "threadpool": [
    "sha256": "909b210d3e892c246a9e610c1f769ca565a42fc5238b2ca72af8f74e254b9c79",
    "sha256" + debug_suffix: "7751c3d298b9d2a1b9a9782a134328a8021049963a54e00d8329366001bb40d5",
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
