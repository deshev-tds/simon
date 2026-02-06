package dev.simon.app.model

enum class ConnectionStatus {
    CONNECTING,
    OPEN,
    CLOSED,
    ERROR,
}

enum class Sender {
    USER,
    AI,
}

data class ImageAttachment(
    val mime: String,
    val dataB64: String,
    val width: Int? = null,
    val height: Int? = null,
    val sizeBytes: Int? = null,
)

data class ChatMessage(
    val id: String,
    val text: String,
    val sender: Sender,
    val timestampMs: Long,
    val isStreaming: Boolean = false,
    val images: List<ImageAttachment> = emptyList(),
)

data class SessionSummary(
    val id: Long,
    val title: String,
    val summary: String? = null,
    val tags: String? = null,
    val model: String? = null,
    val createdAtS: Double,
    val updatedAtS: Double,
)

data class StoredMessage(
    val id: Long,
    val role: String,
    val content: String,
    val createdAtS: Double?,
    val tokens: Int?,
    val attachments: List<ImageAttachment> = emptyList(),
)

data class LiveTranscript(
    val stable: String,
    val draft: String,
    val isFinal: Boolean = false,
)

data class SessionWindow(
    val session: SessionSummary,
    val anchors: List<StoredMessage>,
    val recents: List<StoredMessage>,
)

