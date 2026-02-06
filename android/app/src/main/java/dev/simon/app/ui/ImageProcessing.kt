package dev.simon.app.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Base64
import dev.simon.app.model.ImageAttachment
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.util.UUID
import kotlin.math.max
import kotlin.math.roundToInt

suspend fun processSingleImageUri(
    context: Context,
    uri: Uri,
    maxEdge: Int,
    maxMb: Int,
): PendingImage? = withContext(Dispatchers.IO) {
    val resolver = context.contentResolver

    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    resolver.openInputStream(uri)?.use { input ->
        BitmapFactory.decodeStream(input, null, bounds)
    } ?: return@withContext null

    val outW = bounds.outWidth
    val outH = bounds.outHeight
    if (outW <= 0 || outH <= 0) return@withContext null

    val maxRaw = max(outW, outH)
    var sample = 1
    while (maxRaw / sample > maxEdge) sample *= 2

    val decodeOpts = BitmapFactory.Options().apply {
        inSampleSize = sample
        inPreferredConfig = Bitmap.Config.ARGB_8888
    }

    val decoded = resolver.openInputStream(uri)?.use { input ->
        BitmapFactory.decodeStream(input, null, decodeOpts)
    } ?: return@withContext null

    val scaled = scaleDown(decoded, maxEdge)

    // Compress to JPEG and enforce max size.
    val maxBytes = maxMb * 1024 * 1024
    val (jpegBytes, _) = compressJpeg(scaled, maxBytes)
        ?: return@withContext null

    val b64 = Base64.encodeToString(jpegBytes, Base64.NO_WRAP)
    val att = ImageAttachment(
        mime = "image/jpeg",
        dataB64 = b64,
        width = scaled.width,
        height = scaled.height,
        sizeBytes = jpegBytes.size,
    )

    PendingImage(
        id = UUID.randomUUID().toString(),
        attachment = att,
        bitmap = scaled,
    )
}

private fun scaleDown(bitmap: Bitmap, maxEdge: Int): Bitmap {
    val maxDim = max(bitmap.width, bitmap.height)
    if (maxDim <= maxEdge) return bitmap
    val scale = maxEdge.toFloat() / maxDim.toFloat()
    val newW = max(1, (bitmap.width * scale).roundToInt())
    val newH = max(1, (bitmap.height * scale).roundToInt())
    return Bitmap.createScaledBitmap(bitmap, newW, newH, true)
}

private fun compressJpeg(bitmap: Bitmap, maxBytes: Int): Pair<ByteArray, Int>? {
    val baos = ByteArrayOutputStream()
    val qualities = listOf(85, 70, 60, 50, 40)
    for (q in qualities) {
        baos.reset()
        val ok = bitmap.compress(Bitmap.CompressFormat.JPEG, q, baos)
        if (!ok) continue
        val bytes = baos.toByteArray()
        if (bytes.size <= maxBytes) return bytes to q
    }
    return null
}
