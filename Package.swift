// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251207"
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
    "sha256": "312522b821152c80429b8684ca1f223e1c62ffc1f6e88beadb826120312094ed",
    "sha256" + debug_suffix: "6f4ac1364c5fb21acd4d7a57d3a250049a6816aa8071ff08380caad9ab490f43",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7b6bcc44c5aeab2b49f57ef071c1826ccb1f31dd8ddcf746290a08dcf238e773",
    "sha256" + debug_suffix: "02fd3fdb7217b0985829cf4786109f4fe1dc823245b072431dc273555621f4c9",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d2c38797f923cd792345bf56764a6f19ecdb1576f898d22896fdd13c7169dd08",
    "sha256" + debug_suffix: "03c3a6ab1b28b3e5dd7416b53af525edcea1e9ed339b079a38d04179a632ab2c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "bf60e158ac8147ec2db5ef09d8c978cef3cc7f8b1f030b79c8fb51aa0a736c6e",
    "sha256" + debug_suffix: "f37ca5ab4a79986191ba4cd2620b403c1a81319fc95e1a79e2be1654d78b6f0f",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "b928e3afbee5d7fabdcd495aa7f80f3f10c09ca38a01c31f1f5f433e505f2456",
    "sha256" + debug_suffix: "5dcdfce0fa0dc6303e427d3e1c7e8c4bbea07463fd7f90b25ce6aee219c9b1c4",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "71eeab1faab3c677d61a0c377139fd142c1081e676a32f5738dd01fcf443e985",
    "sha256" + debug_suffix: "04a794161d01747b4a4f44d8fb4e5b684fe9641faf6804aaf0387f45818e8c38",
  ],
  "kernels_optimized": [
    "sha256": "ded0cae5c458e8453a2e3c174b7e4ba241e14c1a28baebbac4c268a3ceb7b5d1",
    "sha256" + debug_suffix: "5cde5b18652a9d78e2160ac1ba57919820df424bfd6d0a3ec649bc95b67eb317",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "fe06ca358a65f6240f35169f61f2a5a4969c7b4dd0c4acd77d3a72356e3a2a8a",
    "sha256" + debug_suffix: "9903bcced2f2473e11d9b765d0db1bae82b93b0efc225d20f3cb4c85412abf86",
  ],
  "kernels_torchao": [
    "sha256": "a393ed2114a176b4fc1f43f049972bb1d6073e605872ba5563e81db63559c5bb",
    "sha256" + debug_suffix: "d25adeafdecc575344903ca9ab42a5a2bb085a9e676da92199a288f96965f5c7",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "739e1ee7057771e52c46bb7dc29b3d804a0431ce28a8d54706b11ed777531542",
    "sha256" + debug_suffix: "94c21ebc2af6ff47999b2d1f75715885768dd58c0d70cf7577d3864498b71a98",
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
