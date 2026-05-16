// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260516"
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
    "sha256": "f8c53ad2c31e019083d10d21edaf566d2036c4bbd374a03db5d8d64062daaada",
    "sha256" + debug_suffix: "3b599287dc112142c57be48bdcebe1ce71677c9c267e05387cf2b173422aa599",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "be15ad35082948103cb29564d7918af62190c995166c3f8917bc7d3b2de27426",
    "sha256" + debug_suffix: "bd3de6ca72845020811cfab4150528f48d9a70c740d1651811878ba8271d9322",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "be317bcd350875194854b666590319cf7cc79bf5d137366e83327f0ff68d43a1",
    "sha256" + debug_suffix: "d39ca1fd3b3bf9926c00256c6411b96ced9c91ae9b502c3163fb56cf826d40e0",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "1f6daaf6fded41e0e958287efcdd1540c11d997e167e1888da19e4dc85b9d85f",
    "sha256" + debug_suffix: "e2499674730f68a9bd10161adcb8fd62f354dccf5790c9f2f5b704bc7ec19a76",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e4dfa0ddf66c8c53cc356eb78b3af7cd19d23d7ecd24dac5c5a8437f9da0a255",
    "sha256" + debug_suffix: "f9c2f20bb86e7a45ac307faab1c0e14df0d5dcb38b529e3a680e0c7078048bc1",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "b2c24513647fc4f9ad50ec4b0a86b4554fe6a89732224ee650b16aa1e5a92d10",
    "sha256" + debug_suffix: "bb53b1d76b21dc438d97066904cdcbd4e409b2c82df2b3832130c82350d0b5c0",
  ],
  "kernels_optimized": [
    "sha256": "25c2804c19088d7f393aa99165086911cda46da2e30d545bae4db26658e92cf8",
    "sha256" + debug_suffix: "33abe6ec4684a4e1e12834501edb0b80eeaabf1973d20c9c660c3a0503b660e3",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9ba9b49fd5a8efc0237331c1aa54f127ebb1d97d10e5af0d078879ae5e24a37b",
    "sha256" + debug_suffix: "27f7b1d99787a79059fbeee1880d708e55dff885ff2c0f11dcc500a12384bdc4",
  ],
  "kernels_torchao": [
    "sha256": "f93cdc9a57853a23abeebb0b13e730954429a9c6cf4db476bc9d9cd46de2b35f",
    "sha256" + debug_suffix: "e87a6848c75471eafc405bedd8f7450a4d72e24646864621285a8faffef3b92d",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "66c95916f8f48942ac7573023c3b161fa71919b3c34605c24300dc92c8217cf2",
    "sha256" + debug_suffix: "9cab690d4ebb6ae9ccadc62045fe6f98c4f0a5510309ef75179aa53be4659697",
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
