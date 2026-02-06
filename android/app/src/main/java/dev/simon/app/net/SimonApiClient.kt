package dev.simon.app.net

import dev.simon.app.model.ImageAttachment
import dev.simon.app.model.SessionSummary
import dev.simon.app.model.SessionWindow
import dev.simon.app.model.StoredMessage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject

data class VisionChatResult(
    val content: String,
    val sessionId: Long?,
)

class SimonApiClient(
    private val client: OkHttpClient,
    private val baseUrl: String,
) {
    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()

    private suspend fun call(request: Request): String = withContext(Dispatchers.IO) {
        client.newCall(request).execute().use { res ->
            val body = res.body?.string().orEmpty()
            if (!res.isSuccessful) {
                throw RuntimeException("HTTP ${res.code}: ${res.message} ${body.take(300)}")
            }
            body
        }
    }

    suspend fun listSessions(): List<SessionSummary> {
        val req = Request.Builder()
            .url("$baseUrl/sessions")
            .get()
            .build()
        val body = call(req)
        val root = JSONObject(body)
        val sessions = root.optJSONArray("sessions") ?: JSONArray()
        return (0 until sessions.length()).mapNotNull { idx ->
            val s = sessions.optJSONObject(idx) ?: return@mapNotNull null
            SessionSummary(
                id = s.optLong("id"),
                title = s.optString("title", ""),
                summary = s.optString("summary", "").takeIf { it.isNotBlank() },
                tags = s.optString("tags", "").takeIf { it.isNotBlank() },
                model = s.optString("model", "").takeIf { it.isNotBlank() },
                createdAtS = s.optDouble("created_at", 0.0),
                updatedAtS = s.optDouble("updated_at", 0.0),
            )
        }
    }

    suspend fun createSession(title: String? = null): SessionSummary {
        val payload = JSONObject()
        if (!title.isNullOrBlank()) payload.put("title", title)

        val req = Request.Builder()
            .url("$baseUrl/sessions")
            .post(payload.toString().toRequestBody(jsonMediaType))
            .build()
        val body = call(req)
        val s = JSONObject(body)
        val now = System.currentTimeMillis() / 1000.0
        return SessionSummary(
            id = s.optLong("id"),
            title = s.optString("title", title ?: ""),
            summary = s.optString("summary", "").takeIf { it.isNotBlank() },
            tags = s.optString("tags", "").takeIf { it.isNotBlank() },
            model = s.optString("model", "").takeIf { it.isNotBlank() },
            createdAtS = now,
            updatedAtS = now,
        )
    }

    suspend fun getSessionWindow(sessionId: Long): SessionWindow {
        val req = Request.Builder()
            .url("$baseUrl/sessions/$sessionId/window")
            .get()
            .build()
        val body = call(req)
        val root = JSONObject(body)
        val sessionObj = root.getJSONObject("session")
        val session = SessionSummary(
            id = sessionObj.optLong("id"),
            title = sessionObj.optString("title", ""),
            summary = sessionObj.optString("summary", "").takeIf { it.isNotBlank() },
            tags = sessionObj.optString("tags", "").takeIf { it.isNotBlank() },
            model = sessionObj.optString("model", "").takeIf { it.isNotBlank() },
            createdAtS = sessionObj.optDouble("created_at", 0.0),
            updatedAtS = sessionObj.optDouble("updated_at", 0.0),
        )

        val anchors = parseStoredMessages(root.optJSONArray("anchors") ?: JSONArray())
        val recents = parseStoredMessages(root.optJSONArray("recents") ?: JSONArray())
        return SessionWindow(
            session = session,
            anchors = anchors,
            recents = recents,
        )
    }

    private fun parseStoredMessages(arr: JSONArray): List<StoredMessage> {
        return (0 until arr.length()).mapNotNull { idx ->
            val m = arr.optJSONObject(idx) ?: return@mapNotNull null
            val atts = m.optJSONArray("attachments")
            val attachments = if (atts != null) parseAttachments(atts) else emptyList()
            StoredMessage(
                id = m.optLong("id"),
                role = m.optString("role", ""),
                content = m.optString("content", ""),
                createdAtS = m.optDouble("created_at").takeIf { !it.isNaN() },
                tokens = m.optInt("tokens").takeIf { it != 0 },
                attachments = attachments,
            )
        }
    }

    private fun parseAttachments(arr: JSONArray): List<ImageAttachment> {
        return (0 until arr.length()).mapNotNull { idx ->
            val a = arr.optJSONObject(idx) ?: return@mapNotNull null
            ImageAttachment(
                mime = a.optString("mime", "image/jpeg"),
                dataB64 = a.optString("data_b64", ""),
                width = a.optInt("width").takeIf { it > 0 },
                height = a.optInt("height").takeIf { it > 0 },
                sizeBytes = a.optInt("size_bytes").takeIf { it > 0 },
            )
        }
    }

    suspend fun visionChat(prompt: String, images: List<ImageAttachment>, sessionId: Long?): VisionChatResult {
        val payload = JSONObject()
        payload.put("prompt", prompt)
        if (sessionId != null) payload.put("session_id", sessionId)
        if (images.isNotEmpty()) {
            val arr = JSONArray()
            for (img in images) {
                val obj = JSONObject()
                obj.put("mime", img.mime)
                obj.put("data_b64", img.dataB64)
                if (img.width != null) obj.put("width", img.width)
                if (img.height != null) obj.put("height", img.height)
                if (img.sizeBytes != null) obj.put("size_bytes", img.sizeBytes)
                arr.put(obj)
            }
            payload.put("images", arr)
        }

        val req = Request.Builder()
            .url("$baseUrl/v1/chat/vision")
            .post(payload.toString().toRequestBody(jsonMediaType))
            .build()
        val body = call(req)
        val root = JSONObject(body)
        val choices = root.optJSONArray("choices")
        val content = if (choices != null && choices.length() > 0) {
            val msg = choices.optJSONObject(0)?.optJSONObject("message")
            msg?.optString("content", "") ?: ""
        } else {
            ""
        }
        val outSessionId = if (root.has("session_id")) root.optLong("session_id") else null
        return VisionChatResult(content = content, sessionId = outSessionId)
    }
}
