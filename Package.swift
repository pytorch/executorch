// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250904"
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
    "sha256": "a7d8ce341aa230f02bc660d77c10bc0a08913ced9051284a92631d7125511d24",
    "sha256" + debug_suffix: "26d8f73ed4c32d938311b5eeac98cb68e4237732b4fca2f992fa61282fed9919",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "f77d29c5f014d1db8851d3407a52e3df03d6a830499adade4e41d20f84d40472",
    "sha256" + debug_suffix: "ddf921c48c27d29a4a24b452a8d6b3552a254194d91ea276f435ce7489262392",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8f8305fabf207cecc69b4e2cf6dd4bca1736f4cb5001911e463e7d0a0649b418",
    "sha256" + debug_suffix: "c6a5a17ed615519e684615240091d8c819e7ae5b830ad6f5f00744d625d0ae56",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e28dc76d156d230e17fe05c219c9bc822006930f430055f37174447e8fef772d",
    "sha256" + debug_suffix: "418f2c0cbb3062e2c27986ce9d8be6e468950ae453e43338adb5079839fb23aa",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "fdeb647991cfc6164c3e9c3607cecaa085d46c9c5fdb0e1ceb1e2c1ce34854de",
    "sha256" + debug_suffix: "52a6a48df59d65cd46e286772f70f294f95cfcd29a344b044d6c4c6e154f7d4d",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "83846cbb606398db3851e0af8f043262255057e9c077e5d2c544f79122e179f8",
    "sha256" + debug_suffix: "7f0bd9e28cc5f13e991cac6711cbf71d7ff29fb8266ed2fdd109e95316c03051",
  ],
  "kernels_optimized": [
    "sha256": "9a12776ca0a8f9ef4070e783408de58c36b8c19e587315a9a6bbb03c489d7fb1",
    "sha256" + debug_suffix: "a5384cc25a6e1341da6e5e3be7854215666a8ae070a1ce95726f61fcea6b2071",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "088cedabec94f4c08e97f35f8341b889ec12e66d4f0b6d1c6e5b44b83c6ca1f3",
    "sha256" + debug_suffix: "11ff5d621a59dc6e16afc08a967f17478cd9646c5b4b27b2deb422fb45710f32",
  ],
  "kernels_torchao": [
    "sha256": "db73518b5e9139f28dc5a78bb24396d85f22c51e79a5f2ee21a200c6cdefbed5",
    "sha256" + debug_suffix: "660ad49fe58e7338d2d903033e30ad16e1cbd1e6bf8df9cefcdf76ed0c85f526",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "c7cf421978cd6e69348760f80278e2e3c1a7696e3afcb658545c59548e61cbfd",
    "sha256" + debug_suffix: "605267d88d941175ed84e6537f0bff59235476d677bb9fb3f6746086abfce325",
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
