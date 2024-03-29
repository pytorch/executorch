// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import Foundation

final class ResourceMonitor: ObservableObject {
  @Published var usedMemory = 0
  @Published var availableMemory = 0
  private var memoryUpdateTimer: Timer?

  deinit {
    stop()
  }

  public func start() {
    memoryUpdateTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
      self?.updateMemoryUsage()
    }
  }

  public func stop() {
    memoryUpdateTimer?.invalidate()
  }

  private func updateMemoryUsage() {
    usedMemory = usedMemoryInMB()
    availableMemory = availableMemoryInMB()
  }

  private func usedMemoryInMB() -> Int {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

    let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
      }
    }
    guard kerr == KERN_SUCCESS else { return 0 }
    return Int(info.resident_size / 0x100000)
  }

  private func availableMemoryInMB() -> Int {
    return Int(os_proc_available_memory() / 0x100000)
  }
}
