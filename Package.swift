// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260107"
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
    "sha256": "3fec270b2c8dd2085563ba4864725eb9a53b8dc86545bcc1b87e4f27377ba26e",
    "sha256" + debug_suffix: "58f3de81dccf07f4d8f36741145eece7615f0ad8701574efb5d5a34d9d057409",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d73716de7ce15cce00068eb6cbe14e57e3ac72c57f3e6725666c03bf75e25bb5",
    "sha256" + debug_suffix: "d4d3a8673290fef6ec57a735a88e9c42c641be6f0e7e18c5c56eac683f3f80ad",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "bf55c8365e6e4cd600be32da8a5136f3e0b01d7d260fc00b894f693aabcb2317",
    "sha256" + debug_suffix: "c4894a05749540782aea569f60b5708c93e5f4f231b52d8874d4fc9731b59802",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "555f67a1b3cdfde8b775f13564b5d0d156a67bd3430e121f7492cae589f517be",
    "sha256" + debug_suffix: "00843e9031db9894bdcfcb9076971bd268c64cb9910eef55d0f41a014a5668ad",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "f5bd7489ed7f320ca242271a76b4924fe7cc2faca69a42e4bc0b9c5e028f581b",
    "sha256" + debug_suffix: "4a5e8f9e03b59199fbf8bec6f5ee6bf1ac7df4eba19ab88ba2860b6e75d8280d",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "75b352a42c9ebe13e8a40606884bf64dbd33fa8b97f6f997cdfa7c4a2dbc0fa2",
    "sha256" + debug_suffix: "02a7925960220d7a3da6a26e94fb57fc2cbca295ec76bd14d6a4cecc80d0dad7",
  ],
  "kernels_optimized": [
    "sha256": "4405947789f107230ea3ee2373f288bb346c976d51113c17fd81c8ef91dd6b3e",
    "sha256" + debug_suffix: "48d7a028e5cf5eed3fd108e5ccaace12788c3d036ffe79688e86bfdd03ea7720",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "07f1bb11529328278f14875adcc7f495d10d6543e738c3d884c1aad711ff9df5",
    "sha256" + debug_suffix: "7a07ec6ffca9402e7d9b9c6c279a4cc2900f085077bc5d093a702251a13d633e",
  ],
  "kernels_torchao": [
    "sha256": "7f72b17d56ec0389754445cd38a72ef02b237e25dac97ada1f5f6da31283b206",
    "sha256" + debug_suffix: "d26668a83b4a488e2cdbe9bc89499c6bbe5524e3ba1ed359707b56d3e5860a3f",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "707a57806a0453353f3806cb32232703e0f443afaf92247907e2519c52fec5a6",
    "sha256" + debug_suffix: "2ecb441676283aac7ad222d666da5b306b474251fbffe04f7f3693973bea51e6",
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
