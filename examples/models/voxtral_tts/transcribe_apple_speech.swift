import Foundation
import Speech

enum TranscriptionError: Error, CustomStringConvertible {
    case badUsage
    case recognizerUnavailable
    case authorizationDenied(Int)
    case recognitionFailed(String)
    case timeout

    var description: String {
        switch self {
        case .badUsage:
            return "usage: swift transcribe_apple_speech.swift <audio_path> [locale]"
        case .recognizerUnavailable:
            return "speech recognizer unavailable"
        case .authorizationDenied(let raw):
            return "speech authorization denied (\(raw))"
        case .recognitionFailed(let message):
            return message
        case .timeout:
            return "speech recognition timed out"
        }
    }
}

func requestAuthorization() throws {
    let semaphore = DispatchSemaphore(value: 0)
    var status = SFSpeechRecognizerAuthorizationStatus.notDetermined
    SFSpeechRecognizer.requestAuthorization { newStatus in
        status = newStatus
        semaphore.signal()
    }
    semaphore.wait()
    guard status == .authorized else {
        throw TranscriptionError.authorizationDenied(status.rawValue)
    }
}

func transcribe(audioPath: String, localeIdentifier: String) throws -> String {
    try requestAuthorization()

    guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: localeIdentifier)) else {
        throw TranscriptionError.recognizerUnavailable
    }

    let request = SFSpeechURLRecognitionRequest(url: URL(fileURLWithPath: audioPath))
    request.shouldReportPartialResults = false

    var finalText: String?
    var finalError: Error?
    var done = false

    let task = recognizer.recognitionTask(with: request) { result, error in
        if let result, result.isFinal {
            finalText = result.bestTranscription.formattedString
            done = true
        }
        if let error {
            finalError = error
            done = true
        }
    }

    let deadline = Date().addingTimeInterval(90)
    while !done && Date() < deadline {
        RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.2))
    }
    task.cancel()

    if let finalText {
        return finalText
    }
    if let finalError {
        throw TranscriptionError.recognitionFailed(String(describing: finalError))
    }
    throw TranscriptionError.timeout
}

do {
    guard CommandLine.arguments.count >= 2 else {
        throw TranscriptionError.badUsage
    }
    let audioPath = CommandLine.arguments[1]
    let locale = CommandLine.arguments.count >= 3 ? CommandLine.arguments[2] : "en-US"
    let transcript = try transcribe(audioPath: audioPath, localeIdentifier: locale)
    print(transcript)
} catch {
    fputs("\(error)\n", stderr)
    exit(1)
}
