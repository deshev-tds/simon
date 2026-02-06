package dev.simon.app.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.material3.MaterialTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

@Composable
fun Base64Image(
    modifier: Modifier,
    mime: String,
    dataB64: String,
) {
    val bmp by produceState<Bitmap?>(initialValue = null, key1 = mime, key2 = dataB64) {
        value = withContext(Dispatchers.IO) {
            decodeBase64Bitmap(dataB64)
        }
    }

    if (bmp != null) {
        Image(
            modifier = modifier,
            bitmap = bmp!!.asImageBitmap(),
            contentDescription = null,
            contentScale = ContentScale.Crop,
        )
    } else {
        Box(modifier = modifier.background(MaterialTheme.colorScheme.surfaceVariant))
    }
}

@Composable
fun BitmapImage(
    modifier: Modifier,
    bitmap: Bitmap,
) {
    Image(
        modifier = modifier,
        bitmap = bitmap.asImageBitmap(),
        contentDescription = null,
        contentScale = ContentScale.Crop,
    )
}

private fun decodeBase64Bitmap(dataB64: String): Bitmap? {
    if (dataB64.isBlank()) return null
    val raw = if (dataB64.startsWith("data:")) {
        dataB64.substringAfter(',', "")
    } else {
        dataB64
    }
    return try {
        val bytes = Base64.decode(raw, Base64.DEFAULT)
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    } catch (_: Throwable) {
        null
    }
}
