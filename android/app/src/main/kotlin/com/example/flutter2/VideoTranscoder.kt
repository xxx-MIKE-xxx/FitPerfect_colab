package <your.package>

import android.content.Context
import android.net.Uri
import androidx.media3.common.MediaItem
import androidx.media3.transformer.TransformationResult
import androidx.media3.transformer.TransformationRequest
import androidx.media3.transformer.Transformer
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File

class VideoTranscoder : FlutterPlugin, MethodChannel.MethodCallHandler {
    private lateinit var channel: MethodChannel
    private lateinit var context: Context

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(binding.binaryMessenger, "video_transcoder")
        channel.setMethodCallHandler(this)
        context = binding.applicationContext
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        if (call.method != "to10fps") { result.notImplemented(); return }
        val src = call.argument<String>("src") ?: run {
            result.error("ARG_ERROR", "Missing src", null); return }
        val dst = call.argument<String>("dst") ?: run {
            result.error("ARG_ERROR", "Missing dst", null); return }

        transcode(Uri.fromFile(File(src)), File(dst), result)
    }

    private fun transcode(src: Uri, dst: File, chanResult: MethodChannel.Result) {
        val req = TransformationRequest.Builder()
            .setFrameRate(10)           // ← 10 fps
            .build()

        val transformer = Transformer.Builder(context)
            .setTransformationRequest(req)
            .addListener(object : Transformer.Listener {
                override fun onTransformationCompleted(
                    output: MediaItem, info: TransformationResult) {
                    chanResult.success(dst.absolutePath)
                }
                override fun onTransformationError(
                    input: MediaItem, e: Throwable, completed: Boolean) {
                    chanResult.error("XCODE_FAIL", e.localizedMessage, null)
                }
            })
            .build()

        transformer.startTransformation(src, Uri.fromFile(dst))
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
    }
}
