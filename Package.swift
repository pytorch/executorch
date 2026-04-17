// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260417"
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
    "sha256": "2131bb9cb80b2e06fb2c0cd006c2bf03ff5585340e975ecda18142a61cfd2031",
    "sha256" + debug_suffix: "9318dab114d83c89e8397ff5fe126a1ebe84b17bf99e4943f59f0744493792b2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4d705b9ff10e98ae06be5f80272d8ded399397226d65d8c81b5083cbb2d6cec5",
    "sha256" + debug_suffix: "a855fdb4b1a47bd39d78a82d764c4d484ea41607ab3745a60a08d798d6ac53c9",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4c5178597781b60764e1542f16f85eb19205414062fcf342e759791a1586e799",
    "sha256" + debug_suffix: "943fc51d2b745b042e300b1ce26786fc8a56822e54321422fe2e4a52f79310a0",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "fbb85d931448ab028202f8611c8d15aaaeadf2d303d9b3ca60092436b64f8e7c",
    "sha256" + debug_suffix: "16804fa89093c53e8087b992b0c8542a988aa15e80c289f9b3764b5257b3a338",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e5cd7010b5fc83ec2108592d664889494e6b4a21ba0fae407c70fc29e54ec1dd",
    "sha256" + debug_suffix: "934547359dfda6bb86e6864b52d5bd7d43a20301493dc26fefa321c29fd86377",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "0905108240b5ec3b1263fa95eefba02e64fb632afe396744276e121f3975a10b",
    "sha256" + debug_suffix: "ff9ebcb78af902fd15cba61079a96b9247ce9d17ec9ea08a85b592df2e9c95bd",
  ],
  "kernels_optimized": [
    "sha256": "c077cf5659f66aff3c9eb1de8b3361528f696564a19e89590b70911ce8256824",
    "sha256" + debug_suffix: "e13ea16a02fc931f94754f5177136c465441a275a4a9833d0caa15888a020966",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9f0131a4d694b3819387833f4d6ae93d84483c05d5ec1d8e50c3ba126a600d52",
    "sha256" + debug_suffix: "7e3404e4d819ce4e0d97c70b9b5166411da3ab45268b2b8779d12543ef938b11",
  ],
  "kernels_torchao": [
    "sha256": "331f3f3fd8335f442700ecd3cada1fd6652687016257ab8bd769f8ecbae62ad8",
    "sha256" + debug_suffix: "d7eb9c5b362323012492ba949c64075c89a0cd9e281fad20e79396b4f2faf692",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "f1db7fdf2aa5c07125e3ae0059149dc604ff0aed62e9678043f52fedda0555f3",
    "sha256" + debug_suffix: "b1c02d95d84ff5316da373272e026035028cd895874793a804b041a1b18795d4",
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
