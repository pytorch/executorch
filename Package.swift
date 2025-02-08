// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250208"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "483b01ca73014e1128bff3e2bff5a9e8bae63deb49b9d25a6062e6d677606034",
    "sha256" + debug: "84a375a4767ebe91599fa90dc2e822123beaad9c8baafccb9bc8f2f36cbf8d94",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "18b9eb40fa956dc119907142e28020ab16e51e8f77e705b01694251f76f8248c",
    "sha256" + debug: "9f3a1cdaf064caa5ece80f47f8ad2f089f7e50c32cc3c07502f7ef3817a74cad",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8d93fa55e7756a85bf729be94b619ef839f35a83372f855b423d1be3ce9a1ace",
    "sha256" + debug: "3159cfe5e6cbbad1ef728406b94df4ad53673e1086f597d16b56099aadc2d33d",
  ],
  "executorch": [
    "sha256": "3e37777204b3ea2f9c7be5607b7dce35fb41df719c34b205fccdaef08b2704cd",
    "sha256" + debug: "9fac57b826cdb2386697db15ca22c98b5a48af2ca61f8b347372975bea920d18",
  ],
  "kernels_custom": [
    "sha256": "a6b3eb32b6f5ef4a3e4a43df566d58741c555e9f64c33d3c748deff7ef7b6983",
    "sha256" + debug: "932742b32ba7ed54f9b3855bc3b64a26aeaecdc929fde83e29534ccca5caf971",
  ],
  "kernels_optimized": [
    "sha256": "60c93b459a9c37ebb275dac92b9fe82624c902392f27995bc71cc48e6464ae01",
    "sha256" + debug: "3bb772c0bd6cd6dee042acedecea4e72f8bda8f99d5bfa83700dfe76d36c3b23",
  ],
  "kernels_portable": [
    "sha256": "3b91a3b39f693c97ba590349574dcb36a1d9375d81c91334b023473f816b108b",
    "sha256" + debug: "a7568b2ab3e3bbb9c74b07b8aec4d1392303848106237e27bb5c02623a1fceb7",
  ],
  "kernels_quantized": [
    "sha256": "b0bc4952fc9a59adb1229fc45da3e03377198e463b7af45ef31d9526648abdf5",
    "sha256" + debug: "688bdb8b961223f95ee2e5449fea51526ff9294a21f3eff5d7f92cee860b1f57",
  ],
].reduce(into: [String: [String: Any]]()) {
  $0[$1.key] = $1.value
  $0[$1.key + debug] = $1.value
}
.reduce(into: [String: [String: Any]]()) {
  var newValue = $1.value
  if $1.key.hasSuffix(debug) {
    $1.value.forEach { key, value in
      if key.hasSuffix(debug) {
        newValue[String(key.dropLast(debug.count))] = value
      }
    }
  }
  $0[$1.key] = newValue.filter { key, _ in !key.hasSuffix(debug) }
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v10_15),
  ],
  products: deliverables.keys.map { key in
    .library(name: key, targets: ["\(key)_dependencies"])
  }.sorted { $0.name < $1.name },
  targets: deliverables.flatMap { key, value -> [Target] in
    [
      .binaryTarget(
        name: key,
        url: "\(url)\(key)-\(version).zip",
        checksum: value["sha256"] as? String ?? ""
      ),
      .target(
        name: "\(key)_dependencies",
        dependencies: [.target(name: key)],
        path: ".swift/\(key)",
        linkerSettings:
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
