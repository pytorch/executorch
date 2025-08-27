// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250827"
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
    "sha256": "a5520e31cf661b965ea5f0c96859d5f54f604c81d4e6f0d126683a08d111fe6c",
    "sha256" + debug_suffix: "248a3059926e97f73651bbbf078fc5c112f6813784a0301b232828cc794bbd41",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "499c0cca7c59459a396389275d26901d66b83fe9335c34204c4e5df90458c310",
    "sha256" + debug_suffix: "ace2f7e6f22b8d8a6ed7d6ea71acf56c87bb023b1d533fbce67f8916e3596948",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e7753e647aad86406f233c9f79d5e499319c71969c3935a4ff54f1376a192f2c",
    "sha256" + debug_suffix: "d716e0353feb5030f1bd303e91994dd835d6697d6dfb56ebb41ca2cc7a44ed3c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "b24e1dc8e1da0132209e10316f1ff3b0791fdd0b4c091ed6fdf5e35126c72089",
    "sha256" + debug_suffix: "1673e5e7448b3b8e4bc7f08ef553acab0735142ad105810af415a74aac47f53d",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "79d25c3a98cda94e6cc758940c7475410512bb1eab491aec5af1a1ba19405b73",
    "sha256" + debug_suffix: "f6e45089bcfcc006e7d788d8c76dd4e2818a885feb2b2c07c30c539ddae94bcf",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "29f9ae6a65218dc070022c2441dbb6c10a64bd93a65db93be7439a1c37dbda63",
    "sha256" + debug_suffix: "89de8684df40dd013590ab4684f9b7c4c02a866f6df88e2a54bad628e0f20c1d",
  ],
  "kernels_optimized": [
    "sha256": "34dcd0531753b17dcca5542af97fae705c761aeae317485d58cae0df7cb1ea9d",
    "sha256" + debug_suffix: "19d7ea02e39707e3fc688dd2d629feaebea6e1926429780f68bff9b14e43d247",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a01ada388f8201ab4671734da4f812f0a412dcd15c87af59fbb9e51ae5940c80",
    "sha256" + debug_suffix: "950275610b95df84479126069db430447616d4b14faba5936e41757d9e6e1bf5",
  ],
  "kernels_torchao": [
    "sha256": "7d5fffbbf265211706c6a146f7cf5443c9e19042f5f94d813ef8d8e0227c327d",
    "sha256" + debug_suffix: "76d0cda0734ee2c2c958af039b934bd5af1472974883f59733b5380828ef7094",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "5e2f204b98a735b7d716d1415bc5b986cdee0ac993556011e02aeeed43075fd8",
    "sha256" + debug_suffix: "38c5ea325cab41cd05d108da280c762d2470d1744d44735715946b8b2210222f",
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
