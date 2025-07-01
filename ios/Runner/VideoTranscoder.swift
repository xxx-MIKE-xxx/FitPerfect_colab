import Flutter
import UIKit
import AVFoundation

public class VideoTranscoder: NSObject, FlutterPlugin {

  // ───────────────────────── plugin boilerplate ────────────────────────────
  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(
        name: "video_transcoder",
        binaryMessenger: registrar.messenger()
    )
    registrar.addMethodCallDelegate(VideoTranscoder(), channel: channel)
  }

  public func handle(_ call: FlutterMethodCall,
                     result: @escaping FlutterResult) {
    guard call.method == "to10fps",
          let args    = call.arguments as? [String: String],
          let srcPath = args["src"],
          let dstPath = args["dst"]  else {
      result(FlutterError(code: "ARG_ERROR",
                          message: "Missing src/dst",
                          details: nil))
      return
    }
    transcode(src: URL(fileURLWithPath: srcPath),
              dst: URL(fileURLWithPath: dstPath),
              result: result)
  }
  // ──────────────────────────────────────────────────────────────────────────

  private func transcode(src: URL,
                         dst: URL,
                         result: @escaping FlutterResult) {

    print("[VideoTranscoder] Swift starting, src: \(src.lastPathComponent)")

    // Clean up any previous file
    try? FileManager.default.removeItem(at: dst)

    let asset = AVAsset(url: src)

    // Build a composition with the original video track
    let comp = AVMutableComposition()
    guard
      let videoTrack = asset.tracks(withMediaType: .video).first,
      let compTrack  = comp.addMutableTrack(withMediaType: .video,
                                            preferredTrackID: kCMPersistentTrackID_Invalid)
    else {
      result(FlutterError(code: "NO_VIDEO",
                          message: "No video track",
                          details: nil))
      return
    }

    try? compTrack.insertTimeRange(
      CMTimeRange(start: .zero, duration: asset.duration),
      of: videoTrack,
      at: .zero
    )

    // ── orientation handling ───────────────────────────────────────────────
    let transform = videoTrack.preferredTransform
    compTrack.preferredTransform = transform   // keep original rotation

    let isPortrait = (abs(transform.b) == 1 && transform.a == 0 &&
                      transform.d == 0)
    let naturalSize = videoTrack.naturalSize
    let renderSize  = isPortrait
        ? CGSize(width: naturalSize.height, height: naturalSize.width)
        : naturalSize

    // ── video composition to clamp fps ─────────────────────────────────────
    let vidComp = AVMutableVideoComposition()
    vidComp.frameDuration = CMTime(value: 1, timescale: 10)   // ← 10 fps
    vidComp.renderSize    = renderSize

    let instruction            = AVMutableVideoCompositionInstruction()
    instruction.timeRange      = CMTimeRange(start: .zero, duration: asset.duration)

    let layerInstruction        = AVMutableVideoCompositionLayerInstruction(assetTrack: compTrack)
    layerInstruction.setTransform(transform, at: .zero)       // keep rotation

    instruction.layerInstructions = [layerInstruction]
    vidComp.instructions          = [instruction]

    // ── export session ─────────────────────────────────────────────────────
    guard let export = AVAssetExportSession(
            asset: comp,
            presetName: AVAssetExportPresetHighestQuality) else {
      result(FlutterError(code: "EXPORT_ERR",
                          message: "Cannot create export session",
                          details: nil))
      return
    }

    export.outputURL                   = dst
    export.outputFileType              = .mp4
    export.videoComposition            = vidComp
    export.timeRange                   = CMTimeRange(start: .zero,
                                                     duration: asset.duration)
    export.shouldOptimizeForNetworkUse = true

    export.exportAsynchronously {
      switch export.status {
      case .completed:
        print("[VideoTranscoder] Swift finished, dst: \(dst.lastPathComponent)")
        result(dst.path)
      default:
        result(FlutterError(code: "EXPORT_FAIL",
                            message: export.error?.localizedDescription,
                            details: nil))
      }
    }
  }
}
