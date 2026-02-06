package dev.simon.app.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.math.min

class PcmAudioRecorder(
    private val sampleRate: Int = 16_000,
) {
    private var record: AudioRecord? = null
    private var job: Job? = null

    val isRecording: Boolean
        get() = job?.isActive == true

    fun start(
        scope: CoroutineScope,
        onChunk: (ByteArray) -> Unit,
        onError: (Throwable) -> Unit,
    ) {
        if (isRecording) return

        val minBuffer = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        ).coerceAtLeast(sampleRate / 2)

        val audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            minBuffer * 2,
        )

        record = audioRecord
        audioRecord.startRecording()

        // Aim for 250ms chunks (matches the web UI).
        val targetChunkBytes = (sampleRate * 2) / 4

        job = scope.launch(Dispatchers.IO) {
            try {
                val readBuf = ByteArray(minBuffer)
                val chunkBuf = ByteArray(targetChunkBytes)
                var chunkOffset = 0

                while (isActive) {
                    val n = audioRecord.read(readBuf, 0, readBuf.size)
                    if (n <= 0) continue

                    var idx = 0
                    while (idx < n) {
                        val toCopy = min(n - idx, targetChunkBytes - chunkOffset)
                        System.arraycopy(readBuf, idx, chunkBuf, chunkOffset, toCopy)
                        chunkOffset += toCopy
                        idx += toCopy

                        if (chunkOffset == targetChunkBytes) {
                            onChunk(chunkBuf.copyOf())
                            chunkOffset = 0
                        }
                    }
                }
            } catch (t: Throwable) {
                onError(t)
            } finally {
                try {
                    audioRecord.stop()
                } catch (_: Throwable) {
                }
                audioRecord.release()
            }
        }
    }

    fun stop() {
        job?.cancel()
        job = null
        record = null
    }
}

