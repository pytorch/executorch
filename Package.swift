// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "latest"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "377cb63f6f3b77609c69ea457f77a87abb17e9cca3af6deb2532f67355c629fa",
    "sha256" + debug: "c9ff91b5e84a21aa20a7d98410dc6323253f038231576e6f2c3146e55aab9794",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "720d7cc729ef86e5caafeb508b945673aa2f54b777e3de598494edaf0dd2254b",
    "sha256" + debug: "89d13bcb8176a681b202943e56cfc04edd108b4b3b289f81272cedcbdf737929",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c35c6428c21ff7751c799a1ad83d3ab99eeceb98e6b6c4e81a9fd3a0ff65f974",
    "sha256" + debug: "1cfb2437026d2e63b3a661ff1c791612afc07eaf4d24d4884fcf0d923c5c4337",
  ],
  "executorch": [
    "sha256": "5701d2e9d9b83a1145f07eac35294d38a7aa5cc43b0a05eb0e73f1e96f07f799",
    "sha256" + debug: "c0e0c77c622581b06396dc7c318f48af34b72a10a2094b8c93cdae4cca8093a5",
  ],
  "kernels_custom": [
    "sha256": "ae4c5676d231af31d2e25f3bfd74090becc459c3456b3fab7f7db10421206acb",
    "sha256" + debug: "aa5767b40f40f123722ccdc063b9660d22a424cc2eda8124a5366f00c2926b36",
  ],
  "kernels_optimized": [
    "sha256": "2a53fe393cd96a64bb2687919ba373220553f1809b143df773f5a6f7001e0176",
    "sha256" + debug: "61c5e31315a7da280aeee3ecf5029fe59d03937f1e08dd8702c15202a5d4fa27",
  ],
  "kernels_portable": [
    "sha256": "fc87c70bc4a85897bffb1a633a2659b2b06da7464b3e7aab9a6354501a6834ba",
    "sha256" + debug: "7fe4b7038e6852c91f7b73bdaac97edafa4d3acfa049bf05f1201d49f3fa0d10",
  ],
  "kernels_quantized": [
    "sha256": "4b8995bfb3f02edb6fa651ecf19d7e633e9f792cf838950c22efe33cf77cbe8e",
    "sha256" + debug: "7eab783b4c26d3488b6766aa1dbdc504b0e74452a55d249eaded8016fdf36b56",
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
    .iOS(.v15),
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
