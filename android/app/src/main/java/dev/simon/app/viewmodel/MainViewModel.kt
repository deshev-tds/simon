package dev.simon.app.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.simon.app.audio.PcmAudioRecorder
import dev.simon.app.audio.WavAudioPlayer
import dev.simon.app.data.AppSettings
import dev.simon.app.data.SettingsStore
import dev.simon.app.model.ChatMessage
import dev.simon.app.model.ConnectionStatus
import dev.simon.app.model.ImageAttachment
import dev.simon.app.model.LiveTranscript
import dev.simon.app.model.Sender
import dev.simon.app.model.SessionSummary
import dev.simon.app.net.PinnedCertificateMismatchException
import dev.simon.app.net.SimonApiClient
import dev.simon.app.net.SimonWsClient
import dev.simon.app.net.VisionChatResult
import dev.simon.app.net.WsEvent
import dev.simon.app.net.buildPinnedOkHttpClient
import dev.simon.app.net.probeServerCertificate
import dev.simon.app.net.TlsCertificateInfo
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

enum class TrustMode {
    FIRST,
    CHANGED,
}

sealed interface AppRoute {
    data object Setup : AppRoute
    data class Trust(
        val mode: TrustMode,
        val certificate: TlsCertificateInfo? = null,
        val oldSha256Hex: String? = null,
        val isLoading: Boolean = false,
        val error: String? = null,
    ) : AppRoute

    data object Main : AppRoute
}

data class UiState(
    val route: AppRoute = AppRoute.Setup,

    val serverHost: String? = null,
    val serverPort: Int = 8000,
    val pinnedCertSha256Hex: String? = null,

    val connectionStatus: ConnectionStatus = ConnectionStatus.CLOSED,
    val sessions: List<SessionSummary> = emptyList(),
    val currentSessionId: Long? = null,
    val currentSessionTitle: String = "New Session",
    val isLoadingSession: Boolean = false,

    val messages: List<ChatMessage> = emptyList(),
    val aiIsSpeaking: Boolean = false,
    val isRecording: Boolean = false,
    val isProcessing: Boolean = false,
    val isAwaitingResponse: Boolean = false,
    val liveTranscript: LiveTranscript? = null,

    val error: String? = null,
)

class MainViewModel(app: Application) : AndroidViewModel(app) {
    private val settingsStore = SettingsStore(app.applicationContext)
    private val recorder = PcmAudioRecorder(sampleRate = 16_000)
    private val player = WavAudioPlayer()

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private var settings: AppSettings? = null
    private var api: SimonApiClient? = null
    private var ws: SimonWsClient? = null
    private var wsJob: Job? = null
    private var reconnectJob: Job? = null

    private var streamingAiId: String? = null
    private var networkKey: String? = null
    private var setupForced: Boolean = false

    init {
        viewModelScope.launch {
            settingsStore.settingsFlow.collect { s ->
                settings = s
                onSettingsChanged(s)
            }
        }
    }

    private fun onSettingsChanged(s: AppSettings) {
        _uiState.update {
            it.copy(
                serverHost = s.serverHost,
                serverPort = s.serverPort,
                pinnedCertSha256Hex = s.pinnedCertSha256Hex,
            )
        }

        val host = s.serverHost?.trim().orEmpty()
        val pinned = s.pinnedCertDerBase64?.trim().orEmpty()
        val currentRoute = _uiState.value.route

        if (host.isBlank()) {
            setupForced = false
            _uiState.update { it.copy(route = AppRoute.Setup) }
            teardownNetwork()
            return
        }

        if (pinned.isBlank()) {
            teardownNetwork()
            if (currentRoute is AppRoute.Trust) {
                val old = currentRoute.oldSha256Hex ?: s.pinnedCertSha256Hex
                _uiState.update { it.copy(route = currentRoute.copy(oldSha256Hex = old)) }
            } else {
                _uiState.update { it.copy(route = AppRoute.Setup) }
            }
            return
        }

        // Pinned cert is available.
        if (currentRoute is AppRoute.Setup && setupForced) {
            // User explicitly opened setup; keep it.
            return
        }
        setupForced = false
        _uiState.update { it.copy(route = AppRoute.Main, error = null) }
        ensureNetworkStack(s)
    }

    fun backToSetup() {
        setupForced = true
        _uiState.update { it.copy(route = AppRoute.Setup) }
        teardownNetwork()
    }

    fun forgetPinnedCertificate() {
        viewModelScope.launch {
            settingsStore.clearPinnedCertificate()
        }
    }

    fun saveServer(host: String, port: Int) {
        val cleaned = sanitizeHostPort(host, port)
        setupForced = false
        viewModelScope.launch {
            settingsStore.setServer(cleaned.first, cleaned.second)
            settingsStore.clearPinnedCertificate()
            settingsStore.setLastSessionId(null)
        }
        teardownNetwork()
        beginTrustFlow(TrustMode.FIRST, host = cleaned.first, port = cleaned.second, oldSha256 = null)
    }

    private fun beginTrustFlow(mode: TrustMode, host: String, port: Int, oldSha256: String?) {
        _uiState.update {
            it.copy(
                route = AppRoute.Trust(mode = mode, isLoading = true, oldSha256Hex = oldSha256),
                error = null,
            )
        }
        viewModelScope.launch {
            try {
                val cert = probeServerCertificate(host, port)
                _uiState.update {
                    it.copy(route = AppRoute.Trust(mode = mode, certificate = cert, oldSha256Hex = oldSha256))
                }
            } catch (t: Throwable) {
                _uiState.update {
                    it.copy(route = AppRoute.Trust(mode = mode, oldSha256Hex = oldSha256, isLoading = false, error = t.message))
                }
            }
        }
    }

    fun trustDisplayedCertificate() {
        val route = _uiState.value.route as? AppRoute.Trust ?: return
        val cert = route.certificate ?: return
        viewModelScope.launch {
            settingsStore.setPinnedCertificate(cert.derBase64, cert.sha256FingerprintHex)
        }
    }

    private fun ensureNetworkStack(s: AppSettings) {
        val host = s.serverHost?.trim().orEmpty()
        val pinnedDer = s.pinnedCertDerBase64?.trim().orEmpty()
        if (host.isBlank() || pinnedDer.isBlank()) return

        val key = "$host:${s.serverPort}:${pinnedDer.hashCode()}"
        if (networkKey == key && api != null && ws != null) {
            return
        }

        teardownNetwork()
        networkKey = key

        val ok = buildPinnedOkHttpClient(expectedHost = host, pinnedDerBase64 = pinnedDer)
        val baseUrl = "https://$host:${s.serverPort}"
        api = SimonApiClient(ok, baseUrl)
        ws = SimonWsClient(ok, "wss://$host:${s.serverPort}/ws")

        wsJob = viewModelScope.launch {
            ws?.events?.collect { ev ->
                handleWsEvent(ev)
            }
        }

        // Kick REST preloads immediately (WS might still be connecting).
        refreshSessions()
        s.lastSessionId?.let { id ->
            loadSessionWindow(id)
            _uiState.update { it.copy(currentSessionId = id) }
        }

        connectWs()
    }

    private fun teardownNetwork() {
        reconnectJob?.cancel()
        reconnectJob = null
        wsJob?.cancel()
        wsJob = null
        ws?.disconnect()
        ws = null
        api = null
        networkKey = null
        streamingAiId = null
        recorder.stop()
        player.stop { speaking -> _uiState.update { it.copy(aiIsSpeaking = speaking) } }
        _uiState.update { it.copy(connectionStatus = ConnectionStatus.CLOSED) }
    }

    private fun connectWs() {
        _uiState.update { it.copy(connectionStatus = ConnectionStatus.CONNECTING) }
        ws?.connect()
    }

    private fun scheduleReconnect() {
        if (_uiState.value.route !is AppRoute.Main) return
        if (reconnectJob?.isActive == true) return
        reconnectJob = viewModelScope.launch {
            delay(2000)
            connectWs()
        }
    }

    private fun handleWsEvent(ev: WsEvent) {
        when (ev) {
            is WsEvent.Opened -> {
                _uiState.update { it.copy(connectionStatus = ConnectionStatus.OPEN) }

                // Negotiate PCM16 voice mode for Android.
                ws?.sendText("AUDIO:PCM16LE:16000")

                // Restore session if available.
                val id = _uiState.value.currentSessionId ?: settings?.lastSessionId
                if (id != null) {
                    ws?.sendText("SESSION:$id")
                }
            }

            is WsEvent.Closed -> {
                _uiState.update {
                    it.copy(
                        connectionStatus = ConnectionStatus.CLOSED,
                        isProcessing = false,
                        isAwaitingResponse = false,
                        liveTranscript = null,
                    )
                }
                streamingAiId = null
                scheduleReconnect()
            }

            is WsEvent.Failure -> {
                _uiState.update {
                    it.copy(
                        connectionStatus = ConnectionStatus.CLOSED,
                        isProcessing = false,
                        isAwaitingResponse = false,
                    )
                }
                streamingAiId = null

                if (isPinnedMismatch(ev.error)) {
                    handleCertChanged()
                    return
                }
                scheduleReconnect()
            }

            is WsEvent.Text -> handleWsText(ev.value)

            is WsEvent.Bytes -> {
                _uiState.update { it.copy(isProcessing = false) }
                player.enqueue(viewModelScope, ev.value) { speaking ->
                    _uiState.update { it.copy(aiIsSpeaking = speaking) }
                }
            }
        }
    }

    private fun handleWsText(text: String) {
        when {
            text == "DONE" -> {
                _uiState.update { it.copy(isProcessing = false, isAwaitingResponse = false) }
                finalizeStreamingMessage()
                refreshSessions()
            }

            text.startsWith("SYS:SESSION:") -> {
                val raw = text.substringAfter("SYS:SESSION:", "").trim()
                val parsed = raw.toLongOrNull()
                if (parsed != null) {
                    _uiState.update { it.copy(currentSessionId = parsed) }
                    viewModelScope.launch { settingsStore.setLastSessionId(parsed) }
                    loadSessionWindow(parsed)
                    refreshSessions()
                }
            }

            text.startsWith("LOG:User:") -> {
                val payload = text.substringAfter("LOG:User:", "").trim()
                if (payload.startsWith("{")) {
                    try {
                        val obj = JSONObject(payload)
                        val msgText = obj.optString("text", "")
                        val images = parseImages(obj.optJSONArray("images"))
                        addUserMessage(msgText, images)
                    } catch (_: Throwable) {
                        addUserMessage(payload, emptyList())
                    }
                } else {
                    addUserMessage(payload, emptyList())
                }
            }

            text.startsWith("STREAM:AI:") -> {
                appendAiDelta(text.substringAfter("STREAM:AI:", ""))
            }

            text.startsWith("LOG:AI:") -> {
                val payload = text.substringAfter("LOG:AI:", "")
                val clean = payload.trimStart()
                finalizeAiMessage(clean)
            }

            text.startsWith("LIVE:STT:") -> {
                try {
                    val obj = JSONObject(text.substringAfter("LIVE:STT:", ""))
                    val event = obj.optString("event", "")
                    when (event) {
                        "reset" -> _uiState.update { it.copy(liveTranscript = null) }
                        "partial" -> _uiState.update {
                            it.copy(
                                liveTranscript = LiveTranscript(
                                    stable = obj.optString("stable", ""),
                                    draft = obj.optString("draft", ""),
                                    isFinal = false,
                                )
                            )
                        }
                        "final" -> _uiState.update {
                            it.copy(
                                liveTranscript = LiveTranscript(
                                    stable = obj.optString("text", ""),
                                    draft = "",
                                    isFinal = true,
                                )
                            )
                        }
                    }
                } catch (_: Throwable) {
                }
            }
        }
    }

    private fun parseImages(arr: JSONArray?): List<ImageAttachment> {
        if (arr == null) return emptyList()
        val out = mutableListOf<ImageAttachment>()
        for (i in 0 until arr.length()) {
            val a = arr.optJSONObject(i) ?: continue
            out.add(
                ImageAttachment(
                    mime = a.optString("mime", "image/jpeg"),
                    dataB64 = a.optString("data_b64", ""),
                    width = a.optInt("width").takeIf { it > 0 },
                    height = a.optInt("height").takeIf { it > 0 },
                    sizeBytes = a.optInt("size_bytes").takeIf { it > 0 },
                )
            )
        }
        return out
    }

    fun refreshSessions() {
        val api = this.api ?: return
        viewModelScope.launch {
            try {
                val items = api.listSessions()
                _uiState.update { it.copy(sessions = items) }
                val current = _uiState.value.currentSessionId
                if (current != null) {
                    val title = items.firstOrNull { it.id == current }?.title?.trim().orEmpty()
                    if (title.isNotBlank()) _uiState.update { it.copy(currentSessionTitle = title) }
                }
            } catch (t: Throwable) {
                if (isPinnedMismatch(t)) {
                    handleCertChanged()
                }
            }
        }
    }

    fun switchSession(sessionId: Long) {
        viewModelScope.launch {
            settingsStore.setLastSessionId(sessionId)
        }
        interrupt()
        streamingAiId = null
        _uiState.update {
            it.copy(
                currentSessionId = sessionId,
                messages = emptyList(),
                isAwaitingResponse = false,
                liveTranscript = null,
            )
        }
        ws?.sendText("SESSION:$sessionId")
        loadSessionWindow(sessionId)
        refreshSessions()
    }

    fun createNewSession() {
        val api = this.api ?: return
        viewModelScope.launch {
            try {
                val s = api.createSession()
                switchSession(s.id)
            } catch (t: Throwable) {
                _uiState.update { it.copy(error = t.message) }
                if (isPinnedMismatch(t)) handleCertChanged()
            }
        }
    }

    private fun loadSessionWindow(sessionId: Long) {
        val api = this.api ?: return
        _uiState.update { it.copy(isLoadingSession = true) }
        viewModelScope.launch {
            try {
                val w = api.getSessionWindow(sessionId)
                val combined = (w.anchors + w.recents).map { sm ->
                    ChatMessage(
                        id = sm.id.toString(),
                        text = sm.content,
                        sender = if (sm.role == "assistant") Sender.AI else Sender.USER,
                        timestampMs = ((sm.createdAtS ?: (System.currentTimeMillis() / 1000.0)) * 1000.0).toLong(),
                        images = sm.attachments,
                    )
                }
                streamingAiId = null
                _uiState.update {
                    it.copy(
                        messages = combined,
                        isAwaitingResponse = false,
                        currentSessionTitle = w.session.title.trim().ifBlank { it.currentSessionTitle },
                    )
                }
            } catch (t: Throwable) {
                _uiState.update { it.copy(error = "Failed to load session window: ${t.message}") }
                if (isPinnedMismatch(t)) handleCertChanged()
            } finally {
                _uiState.update { it.copy(isLoadingSession = false) }
            }
        }
    }

    fun sendMessage(text: String, images: List<ImageAttachment>) {
        val trimmed = text.trim()
        if (trimmed.isBlank() && images.isEmpty()) return

        val ws = this.ws
        if (ws != null && ws.isOpen()) {
            if (images.isNotEmpty()) {
                val obj = JSONObject()
                obj.put("type", "chat")
                obj.put("prompt", trimmed)
                val arr = JSONArray()
                for (img in images) {
                    val a = JSONObject()
                    a.put("mime", img.mime)
                    a.put("data_b64", img.dataB64)
                    if (img.width != null) a.put("width", img.width)
                    if (img.height != null) a.put("height", img.height)
                    if (img.sizeBytes != null) a.put("size_bytes", img.sizeBytes)
                    arr.put(a)
                }
                obj.put("images", arr)
                ws.sendText(obj.toString())
            } else {
                ws.sendText(trimmed)
            }
            _uiState.update { it.copy(isAwaitingResponse = true) }
            return
        }

        // REST fallback (still TLS-only). Optimistically render the user message.
        addUserMessage(trimmed, images)
        _uiState.update { it.copy(isAwaitingResponse = true) }
        val api = this.api ?: return
        val currentSessionId = _uiState.value.currentSessionId
        viewModelScope.launch {
            try {
                val res: VisionChatResult = api.visionChat(trimmed, images, currentSessionId)
                finalizeAiMessage(res.content)
                if (res.sessionId != null) {
                    settingsStore.setLastSessionId(res.sessionId)
                    _uiState.update { it.copy(currentSessionId = res.sessionId) }
                    loadSessionWindow(res.sessionId)
                } else {
                    refreshSessions()
                }
            } catch (t: Throwable) {
                finalizeAiMessage("Error: failed to reach server.")
                if (isPinnedMismatch(t)) handleCertChanged()
            } finally {
                _uiState.update { it.copy(isAwaitingResponse = false) }
            }
        }
    }

    fun startRecording() {
        val ws = this.ws
        if (ws == null || !ws.isOpen()) {
            _uiState.update { it.copy(error = "WebSocket is not connected.") }
            return
        }

        // Reset audio + transcript state.
        player.stop { speaking -> _uiState.update { it.copy(aiIsSpeaking = speaking) } }
        _uiState.update { it.copy(aiIsSpeaking = false, isProcessing = false, isAwaitingResponse = false, liveTranscript = null) }

        recorder.start(
            scope = viewModelScope,
            onChunk = { bytes -> ws.sendBytes(bytes) },
            onError = { t ->
                _uiState.update { it.copy(error = "Recording error: ${t.message}", isRecording = false) }
            },
        )
        _uiState.update { it.copy(isRecording = true, error = null) }
    }

    fun stopRecordingAndCommit() {
        val ws = this.ws
        if (ws == null || !ws.isOpen()) return
        recorder.stop()
        _uiState.update { it.copy(isRecording = false) }
        ws.sendText("CMD:COMMIT_AUDIO")
        _uiState.update { it.copy(isProcessing = true, isAwaitingResponse = true) }
    }

    fun interrupt() {
        recorder.stop()
        ws?.sendText("STOP")
        player.stop { speaking -> _uiState.update { it.copy(aiIsSpeaking = speaking) } }
        streamingAiId = null
        _uiState.update {
            it.copy(
                aiIsSpeaking = false,
                isProcessing = false,
                isAwaitingResponse = false,
                isRecording = false,
                liveTranscript = null,
            )
        }
    }

    private fun addUserMessage(text: String, images: List<ImageAttachment>) {
        _uiState.update { st ->
            st.copy(
                messages = st.messages + ChatMessage(
                    id = UUID.randomUUID().toString(),
                    text = text,
                    sender = Sender.USER,
                    timestampMs = System.currentTimeMillis(),
                    images = images,
                )
            )
        }
    }

    private fun appendAiDelta(delta: String) {
        if (delta.isEmpty()) return
        _uiState.update { it.copy(isAwaitingResponse = false) }

        val activeId = streamingAiId
        if (activeId == null) {
            val id = UUID.randomUUID().toString()
            streamingAiId = id
            _uiState.update { st ->
                st.copy(
                    messages = st.messages + ChatMessage(
                        id = id,
                        text = delta,
                        sender = Sender.AI,
                        timestampMs = System.currentTimeMillis(),
                        isStreaming = true,
                    )
                )
            }
            return
        }

        _uiState.update { st ->
            val updated = st.messages.map { m ->
                if (m.id == activeId) m.copy(text = m.text + delta, isStreaming = true) else m
            }
            st.copy(messages = updated)
        }
    }

    private fun finalizeStreamingMessage() {
        val activeId = streamingAiId ?: return
        _uiState.update { st ->
            st.copy(messages = st.messages.map { m ->
                if (m.id == activeId) m.copy(isStreaming = false) else m
            })
        }
        streamingAiId = null
    }

    private fun finalizeAiMessage(text: String) {
        _uiState.update { it.copy(isAwaitingResponse = false) }
        val activeId = streamingAiId
        if (activeId == null) {
            _uiState.update { st ->
                st.copy(
                    messages = st.messages + ChatMessage(
                        id = UUID.randomUUID().toString(),
                        text = text,
                        sender = Sender.AI,
                        timestampMs = System.currentTimeMillis(),
                    )
                )
            }
            return
        }
        _uiState.update { st ->
            st.copy(messages = st.messages.map { m ->
                if (m.id == activeId) m.copy(text = text, isStreaming = false) else m
            })
        }
        streamingAiId = null
    }

    private fun isPinnedMismatch(t: Throwable): Boolean {
        var cur: Throwable? = t
        while (cur != null) {
            if (cur is PinnedCertificateMismatchException) return true
            cur = cur.cause
        }
        return false
    }

    private fun handleCertChanged() {
        val s = settings ?: return
        val host = s.serverHost?.trim().orEmpty()
        if (host.isBlank()) return
        val port = s.serverPort
        val old = s.pinnedCertSha256Hex
        // Force re-trust.
        viewModelScope.launch { settingsStore.clearPinnedCertificate() }
        teardownNetwork()
        beginTrustFlow(TrustMode.CHANGED, host = host, port = port, oldSha256 = old)
    }

    private fun sanitizeHostPort(rawHost: String, rawPort: Int): Pair<String, Int> {
        var h = rawHost.trim()
        h = h.removePrefix("https://").removePrefix("http://")
        h = h.substringBefore("/").substringBefore("?").substringBefore("#")

        var p = rawPort
        val colonIdx = h.lastIndexOf(':')
        if (colonIdx > 0 && colonIdx < h.length - 1) {
            val maybePort = h.substring(colonIdx + 1).toIntOrNull()
            val maybeHost = h.substring(0, colonIdx)
            if (maybePort != null && maybePort in 1..65535) {
                h = maybeHost
                if (p !in 1..65535 || p == 8000) p = maybePort
            }
        }

        if (p !in 1..65535) p = 8000
        return h to p
    }
}
