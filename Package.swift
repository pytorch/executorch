// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251205"
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
    "sha256": "67d2e8fcc4ef8df692f17eca543fe8a10fc6f8e2882edcef26d2ac0365dc364c",
    "sha256" + debug_suffix: "e8518b21e0a024aa3999aa2849559b5cd768b46909aa4985fc7f4bd25ce76423",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8254501ef1f27a85de4bb88fe63430da73565c75a1e8ce8af6d12fba9bc78bc2",
    "sha256" + debug_suffix: "7a89a316229b01472f2e45ef66db15156571bbd40848105c10e892f6d7f4cbf9",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "32bd89286c6a2a98c041a03392c33146ebd0297114205c02a0374dacd138c2dc",
    "sha256" + debug_suffix: "b6ad8a5ee9130182ffad19af1ce532533d506339168a2d6835f0a0fa928fb56f",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "966bdc3ec810e3746938b7abda8b818ef49ec493ae6b7caaffdb18b34ba11ef9",
    "sha256" + debug_suffix: "41a8a2fc8183607f25536252d0943c2df2f678358a1472c0c1ff71dc0ef16b4b",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "3b2fd003167d461fcc22a4763e47f7881e164e8886bb060a77e9c3468f04fa7d",
    "sha256" + debug_suffix: "9768180de893ec93da5c75edb7eceeaaa15eb33c84355f554dda556ef30f3f0a",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "a94d8358ad98883fe91910ec9b8b5ea69f3bad472480e7134e63b5078eb831c1",
    "sha256" + debug_suffix: "44ac6f46ea015867415f322a2ec959381438be0643be4a4f59c8f77218ed98ac",
  ],
  "kernels_optimized": [
    "sha256": "fe76c667a7bb9550a50f6eecd9f0cc357e9df1536b69607cc6b15d9e2901d20a",
    "sha256" + debug_suffix: "c03e693388966171766c6eb35e9f1007e927302f789da6d7ddb38d43e2cae6cf",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "132f857e7f679907bd04f875d04d6ca40f326e698f1b48c951b7611e04df1b89",
    "sha256" + debug_suffix: "aedc9668458ae23aa5b77fc63c5f06d782547b3e791f6a6647e0911f6a317810",
  ],
  "kernels_torchao": [
    "sha256": "2f876d19fee9631b9d4c97f84d522aa49cf8e355f16185b1516b7c3493e31523",
    "sha256" + debug_suffix: "acd3cb61b4f85f0cf0d702697d158bf382a4b2e6c393732e070ce25c15c1fbfa",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a555d5fc2c1c9a738f69d9ccceed0384c908e584a08f833daa4b7241f2ac776b",
    "sha256" + debug_suffix: "ab35397d3e0ede3b2926e09b14087b7e450920fa3e424d84ffa9d2619af56935",
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
