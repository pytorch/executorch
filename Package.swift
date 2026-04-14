// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260414"
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
    "sha256": "a1da26b9a86066a61131374529ef61ead18f3f3ef14c5c27f3ac9cfc400270cc",
    "sha256" + debug_suffix: "852ea7f50bcba1342e6425af87bc3095a1246de655828e24971092f85adf9e8b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "557c67d8b3b49db4f9a193c3a5d56d495e6e77dcbd1cd476779d073ef80235e1",
    "sha256" + debug_suffix: "21ee6944ea7c3f1df92d3c866ee510f6db7937430fd5a7c768b94ba5bc3eadcc",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "857dfafbeba5612812ec163ba29cf0aa203fca51b8e985a741d921b4a570c117",
    "sha256" + debug_suffix: "871a5d1b7f8b777b9dcec9454852cad61291ab168e8ce10de5212d730bea0f1b",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "916daed63c115fa7950b42b64c56b2045fa6bcffe1d84c75cafdea00423a598f",
    "sha256" + debug_suffix: "dfa1b52cbdc78603295158c568efb9699c5adfe992dfda8d2f415d9f82512e3c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "063a2f7f7ec92b0d09844a8ddb7b835e5a4cda6da264f2e67f007ef09058400e",
    "sha256" + debug_suffix: "c629ea070df4b2f46585d22f1a8d558bd92d0cac82f440e7681fc8c02bef23fa",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "a20ffc7e96b9fd4f89764a07f39b82c234664c2b288b9c00b052da641c7b7ba1",
    "sha256" + debug_suffix: "34ed2f0d053a5e35c9fac0bfc67e9c3e4f57164d810ebc8cc1706e87fdaacc2c",
  ],
  "kernels_optimized": [
    "sha256": "ee68a3299f4bc20d3fe945e843ce66a7fbdd6d1f3aac5022043027dfddc49dc6",
    "sha256" + debug_suffix: "8a26437e027d99a53800e038703719d2e4b5ab9f0fff29f258fc6f4f6f57f0d3",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "0e9db959300eb08f75dab1c66d475563c8e2441b3df2ac1433732a7812decd76",
    "sha256" + debug_suffix: "db441c7adcecba2033b2e593af450d8d310ec6621c0f720ccd9637b3a44cb728",
  ],
  "kernels_torchao": [
    "sha256": "bfcc08525205c82a2043218553db5e4cd85f96b14ab2b677e2fac9e313a6ec2a",
    "sha256" + debug_suffix: "2710ea94835df3507b79d300a36c561c11f02a7fa841d74faf1ed0407e6b894e",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "fa93d92a86d8ae56cd0ef757d53c5514b59d20744166872e96021a8ea179ec1d",
    "sha256" + debug_suffix: "040d9333d9e445bae03a13f39c42b9f1b974e6659676b0caa55d7457d2617f1a",
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
