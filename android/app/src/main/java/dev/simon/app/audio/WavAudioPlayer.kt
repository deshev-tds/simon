package dev.simon.app.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min

data class WavData(
    val sampleRate: Int,
    val channels: Int,
    val bitsPerSample: Int,
    val pcm: ByteArray,
)

private fun parseWav(wav: ByteArray): WavData? {
    if (wav.size < 44) return null
    if (!(wav[0].toInt().toChar() == 'R' && wav[1].toInt().toChar() == 'I' && wav[2].toInt().toChar() == 'F' && wav[3].toInt().toChar() == 'F')) return null
    if (!(wav[8].toInt().toChar() == 'W' && wav[9].toInt().toChar() == 'A' && wav[10].toInt().toChar() == 'V' && wav[11].toInt().toChar() == 'E')) return null

    val bb = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN)
    var pos = 12

    var sampleRate: Int? = null
    var channels: Int? = null
    var bitsPerSample: Int? = null
    var pcmStart: Int? = null
    var pcmLen: Int? = null

    while (pos + 8 <= wav.size) {
        val chunkId = String(wav, pos, 4, Charsets.US_ASCII)
        val chunkSize = bb.getInt(pos + 4)
        val dataPos = pos + 8
        if (dataPos + chunkSize > wav.size) break

        if (chunkId == "fmt ") {
            if (chunkSize >= 16) {
                val audioFormat = bb.getShort(dataPos).toInt() and 0xFFFF
                val ch = bb.getShort(dataPos + 2).toInt() and 0xFFFF
                val sr = bb.getInt(dataPos + 4)
                val bps = bb.getShort(dataPos + 14).toInt() and 0xFFFF

                // Expect PCM (1). Backend uses soundfile -> PCM16.
                if (audioFormat != 1) return null
                sampleRate = sr
                channels = ch
                bitsPerSample = bps
            }
        } else if (chunkId == "data") {
            pcmStart = dataPos
            pcmLen = chunkSize
        }

        pos = dataPos + chunkSize + (chunkSize and 1)
        if (sampleRate != null && pcmStart != null && pcmLen != null && channels != null && bitsPerSample != null) break
    }

    val sr = sampleRate ?: return null
    val ch = channels ?: return null
    val bps = bitsPerSample ?: return null
    val ds = pcmStart ?: return null
    val dl = pcmLen ?: return null
    if (ds + dl > wav.size) return null

    val pcm = wav.copyOfRange(ds, ds + dl)
    return WavData(sampleRate = sr, channels = ch, bitsPerSample = bps, pcm = pcm)
}

class WavAudioPlayer {
    private val queue = Channel<ByteArray>(capacity = 64)
    private var job: Job? = null
    private var track: AudioTrack? = null
    private var currentSampleRate: Int? = null
    private var currentChannels: Int? = null

    @Volatile
    var isSpeaking: Boolean = false
        private set

    fun enqueue(scope: CoroutineScope, wavBytes: ByteArray, onSpeakingChanged: (Boolean) -> Unit) {
        if (job == null) {
            job = scope.launch(Dispatchers.IO) { runLoop(onSpeakingChanged) }
        }
        queue.trySend(wavBytes)
    }

    fun stop(onSpeakingChanged: (Boolean) -> Unit) {
        job?.cancel()
        job = null
        drainQueue()
        isSpeaking = false
        onSpeakingChanged(false)
        try {
            track?.pause()
            track?.flush()
            track?.stop()
        } catch (_: Throwable) {
        }
        try {
            track?.release()
        } catch (_: Throwable) {
        }
        track = null
        currentSampleRate = null
        currentChannels = null
    }

    private fun drainQueue() {
        while (true) {
            val polled = queue.tryReceive().getOrNull() ?: break
            @Suppress("UNUSED_VARIABLE")
            val _ignored = polled
        }
    }

    private suspend fun runLoop(onSpeakingChanged: (Boolean) -> Unit) {
        isSpeaking = false
        onSpeakingChanged(false)

        try {
            while (true) {
                val wav = queue.receive()
                val parsed = parseWav(wav) ?: continue
                ensureTrack(parsed)

                if (!isSpeaking) {
                    isSpeaking = true
                    onSpeakingChanged(true)
                }

                val t = track ?: continue
                if (t.playState != AudioTrack.PLAYSTATE_PLAYING) t.play()

                // Stream PCM into AudioTrack.
                var offset = 0
                while (offset < parsed.pcm.size) {
                    val toWrite = min(parsed.pcm.size - offset, 8192)
                    val written = t.write(parsed.pcm, offset, toWrite)
                    if (written <= 0) break
                    offset += written
                }

                // If nothing arrives shortly after finishing a chunk, consider speech done.
                val next = withTimeoutOrNull(200) { queue.receive() }
                if (next == null) {
                    if (isSpeaking) {
                        isSpeaking = false
                        onSpeakingChanged(false)
                    }
                } else {
                    queue.trySend(next)
                }
            }
        } finally {
            withContext(Dispatchers.IO) {
                try {
                    track?.pause()
                    track?.flush()
                    track?.stop()
                } catch (_: Throwable) {
                }
                try {
                    track?.release()
                } catch (_: Throwable) {
                }
                track = null
                currentSampleRate = null
                currentChannels = null
            }
        }
    }

    private fun ensureTrack(parsed: WavData) {
        val sr = parsed.sampleRate
        val ch = parsed.channels
        if (track != null && currentSampleRate == sr && currentChannels == ch) return

        try {
            track?.release()
        } catch (_: Throwable) {
        }

        val channelMask = when (ch) {
            1 -> AudioFormat.CHANNEL_OUT_MONO
            2 -> AudioFormat.CHANNEL_OUT_STEREO
            else -> AudioFormat.CHANNEL_OUT_MONO
        }
        val encoding = when (parsed.bitsPerSample) {
            16 -> AudioFormat.ENCODING_PCM_16BIT
            else -> AudioFormat.ENCODING_PCM_16BIT
        }

        val minBuffer = AudioTrack.getMinBufferSize(sr, channelMask, encoding).coerceAtLeast(8192)
        val audioFormat = AudioFormat.Builder()
            .setSampleRate(sr)
            .setEncoding(encoding)
            .setChannelMask(channelMask)
            .build()
        val attrs = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .build()
        track = AudioTrack(
            attrs,
            audioFormat,
            minBuffer,
            AudioTrack.MODE_STREAM,
            AudioManager.AUDIO_SESSION_ID_GENERATE,
        )
        currentSampleRate = sr
        currentChannels = ch
    }
}

