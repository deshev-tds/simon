package dev.simon.app.net

import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import okio.ByteString.Companion.toByteString

sealed interface WsEvent {
    data object Opened : WsEvent
    data class Closed(val code: Int, val reason: String) : WsEvent
    data class Text(val value: String) : WsEvent
    data class Bytes(val value: ByteArray) : WsEvent
    data class Failure(val error: Throwable) : WsEvent
}

class SimonWsClient(
    private val client: OkHttpClient,
    private val wsUrl: String,
) {
    private val _events = MutableSharedFlow<WsEvent>(
        extraBufferCapacity = 128,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    val events: SharedFlow<WsEvent> = _events.asSharedFlow()

    @Volatile
    private var socket: WebSocket? = null

    fun connect() {
        if (socket != null) return
        val req = Request.Builder().url(wsUrl).build()
        socket = client.newWebSocket(req, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                _events.tryEmit(WsEvent.Opened)
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                _events.tryEmit(WsEvent.Text(text))
            }

            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                _events.tryEmit(WsEvent.Bytes(bytes.toByteArray()))
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                webSocket.close(code, reason)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                socket = null
                _events.tryEmit(WsEvent.Closed(code, reason))
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                socket = null
                _events.tryEmit(WsEvent.Failure(t))
            }
        })
    }

    fun disconnect(code: Int = 1000, reason: String = "client_close") {
        socket?.close(code, reason)
        socket = null
    }

    fun sendText(text: String): Boolean {
        return socket?.send(text) ?: false
    }

    fun sendBytes(bytes: ByteArray): Boolean {
        return socket?.send(bytes.toByteString()) ?: false
    }

    fun isOpen(): Boolean = socket != null
}
