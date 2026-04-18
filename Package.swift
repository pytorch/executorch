// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260418"
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
    "sha256": "2242cab745ad3133e503d1e52377c898f19b6f043625afbee3604c263dfc9b1b",
    "sha256" + debug_suffix: "432f131ca1542b4b46e1dac1cc02fb8ca1403be4210fcdd3d5bf136874aa2f89",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8727dee738f1a13954e34fbf99298e07f57158374c95d9f122a3befb23f9a91a",
    "sha256" + debug_suffix: "b52df8db06d8293aa56b81dfa11215bb33a5bec04b6056476609363f63cb0c9f",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1f6fb2aef508f99c6939ec72d8b27955be88a813aede36a23eaa05bdd8dfe9f4",
    "sha256" + debug_suffix: "355f48b73e6200f473fb4d902e57f832f60a9045d0b8fcc99ea5d1ac888e5003",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "10c03f59d814d1a650b9ef02b3b8f1f468723166b89fc15c0abdcfc4e037f893",
    "sha256" + debug_suffix: "d757a485b110eee71d46ccd722427d03f599f0d1a42284b7ffe6153fee219122",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "3a86acb4d947704dbda3caa606684e8243cfebefe5e29cb8232e682bcd9a2b86",
    "sha256" + debug_suffix: "370b17aab156ecbfb49bd2550003778a9ddd294a3e3af3d27b836509330c2c61",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "bc5f42720058c689d2cac0f1d925e9be3bd9e3fc994c12f3072ede5108142e0a",
    "sha256" + debug_suffix: "61b0a653f9ee086a0bc4e0290eb84408caf46e722cf91d4713c7928134e599c6",
  ],
  "kernels_optimized": [
    "sha256": "ce3376b3c05584b1638b5f88b7a9f60c6ec2249f66909a5ebdd53d6972a8fd08",
    "sha256" + debug_suffix: "5f43e454de061288f2ab00a4f890bef4ae1d72d13a81b665d97d9c7397f8499a",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b41c568d22b6469d33c161d3a0a4039dd8c8baee564056de6ba2b3e9caa11490",
    "sha256" + debug_suffix: "2ea1c87c89c92e7bad9340b7ce50fd66ac2c9c3a270e3d35dfc82acdc4812754",
  ],
  "kernels_torchao": [
    "sha256": "bc14ff970b127c42fced6dbba22abc2db7bb34687cc4f315064ee432d7074f52",
    "sha256" + debug_suffix: "56ec292a4f7b4f4634fb88ecf7db0f904dd15a0314e4c1cef35115bcf394475a",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a97e95a84da8cc3dac2071db547cfbf3b368dceda7e54ca1bf7a03b1b218c0ef",
    "sha256" + debug_suffix: "8acc529930c4afeff755be19813b145ae866a12080024ee8a1e264379ff23f03",
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

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v12),
  ],
  products: packageProducts,
  targets: packageTargets
)
