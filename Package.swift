// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260426"
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
    "sha256": "f332fd11cb4d2dee9f9549adf6478fa0524558a0b9cba67b0811637996cf34ad",
    "sha256" + debug_suffix: "f6956a116cc84aa3b147b2742fc206540863126d165c608c0d6a627425eb14e1",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4b6e21431677a20010445dd602d9eee298862124df65212d06d68acda54373eb",
    "sha256" + debug_suffix: "2d29670fcb8ceae9236019d06ae1aff2dfa496aacc896970f9cdf56ef10edcf5",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a6391a2f02b5e4ae3f18dfa225adddbc18ed735447e5221694fa48f9ce3cc909",
    "sha256" + debug_suffix: "a5ca08184953413857c3da7028ba6f01e9c2a7c19c3dd6b6837f76ec1f987ada",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "8b4d92e1d71c2e595986e84954d13ca788af53e51e76d0e34a2fbf63c45b5843",
    "sha256" + debug_suffix: "0412484dcd8933e84bd0e822080e22e1c323dabe203171ac7d7642644ab7812e",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "002db8716694f5d1ded652587db5d5f99dbbb3001b10f8fe1bf6d33dbc4b791d",
    "sha256" + debug_suffix: "e934379e908490a6c45ff552c3ef1a5340fd1c949145d74396fee4fd21185b32",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "6bf159ccd7212b0321a2ab81b924397a9f6199eed1a54be74b148b5d369a81bc",
    "sha256" + debug_suffix: "72dccc9a12c28e30e7c71bf63155a3aabde9b01bc6fca108e9accb41bbe5013c",
  ],
  "kernels_optimized": [
    "sha256": "c6033837f1139fd78a61ddc131252b4db29bb20fd0ab0c2779bca97b963f6ae4",
    "sha256" + debug_suffix: "1cc5ce3347860c12fa55de48a18ea4cf31cb6064fa1d3acb2a55b6634687500f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "0d542830abd866569f9f37c7c2eb5f34cfdce0cda6c6cc72c6b0c48270a31707",
    "sha256" + debug_suffix: "b9f8a9c4f70fe0076341100636be8609f66709e8dcac6d86ef23fbdbb220da96",
  ],
  "kernels_torchao": [
    "sha256": "91beb478bedf6023c1350a06458fd7bb3cce1a0a780dc53080a79eed730166c6",
    "sha256" + debug_suffix: "c0426716750b2e4c54b1a782e2e1589df034fcebf0ce3ced455ee13b3b99c780",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e6bacee13ed75522db263e743d4a1a478e9e0d25496a2f3aaf0c13281b12ac02",
    "sha256" + debug_suffix: "638a2e438ca5beba8411768e44d35061bb2325b5795ea7d3e97e57ac2e03f9d7",
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
