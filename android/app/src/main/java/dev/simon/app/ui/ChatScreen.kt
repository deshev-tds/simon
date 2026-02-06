package dev.simon.app.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import dev.simon.app.model.ChatMessage
import dev.simon.app.model.ConnectionStatus
import dev.simon.app.model.ImageAttachment
import dev.simon.app.model.Sender
import dev.simon.app.viewmodel.UiState
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    state: UiState,
    onOpenDrawer: () -> Unit,
    onSend: (text: String, images: List<ImageAttachment>) -> Unit,
    onOpenVoice: () -> Unit,
) {
    val listState = rememberLazyListState()
    val scope = rememberCoroutineScope()
    val atBottom by remember { derivedStateOf { !listState.canScrollForward } }
    val context = LocalContext.current

    var inputText by remember { mutableStateOf("") }
    var pendingImages by remember { mutableStateOf<List<PendingImage>>(emptyList()) }
    var imageError by remember { mutableStateOf<String?>(null) }

    val pickImagesLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickMultipleVisualMedia(10),
        onResult = { uris ->
            if (uris.isNullOrEmpty()) return@rememberLauncherForActivityResult
            scope.launch {
                val result = processPickedImages(context, uris, maxEdge = 1024, maxMb = 8)
                val combined = (pendingImages + result.images).take(10)
                pendingImages = combined
                imageError = result.error
            }
        }
    )

    LaunchedEffect(state.messages.size) {
        if (atBottom && state.messages.isNotEmpty()) {
            listState.animateScrollToItem(state.messages.size - 1)
        }
    }

    val online = state.connectionStatus == ConnectionStatus.OPEN
    val title = state.currentSessionTitle.ifBlank { "New Session" }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("NEURAL LINK", style = MaterialTheme.typography.labelSmall)
                        Text(
                            text = "${if (online) "Online" else "Offline"} | $title",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onOpenDrawer) {
                        Icon(Icons.Default.Menu, contentDescription = "Menu")
                    }
                },
            )
        },
        floatingActionButton = {
            if (!atBottom && state.messages.isNotEmpty()) {
                FloatingActionButton(onClick = {
                    scope.launch { listState.animateScrollToItem(state.messages.size - 1) }
                }) {
                    Text("Down")
                }
            }
        },
        bottomBar = {
            Column(modifier = Modifier.fillMaxWidth().padding(12.dp)) {
                if (pendingImages.isNotEmpty()) {
                    PendingImagesRow(
                        images = pendingImages,
                        onRemove = { id -> pendingImages = pendingImages.filterNot { it.id == id } },
                    )
                    Spacer(Modifier.height(8.dp))
                }
                if (!imageError.isNullOrBlank()) {
                    Text(
                        text = imageError ?: "",
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Spacer(Modifier.height(8.dp))
                }
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    IconButton(onClick = onOpenVoice) {
                        Icon(Icons.Default.Mic, contentDescription = "Voice")
                    }
                    IconButton(onClick = {
                        imageError = null
                        pickImagesLauncher.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                        )
                    }) {
                        Icon(Icons.Default.AddPhotoAlternate, contentDescription = "Attach images")
                    }

                    OutlinedTextField(
                        modifier = Modifier.weight(1f),
                        value = inputText,
                        onValueChange = { inputText = it },
                        singleLine = true,
                        placeholder = { Text("Transmit message...") },
                    )

                    IconButton(
                        onClick = {
                            val images = pendingImages.map { it.attachment }
                            val text = inputText.trim()
                            if (text.isBlank() && images.isEmpty()) return@IconButton
                            onSend(text, images)
                            inputText = ""
                            pendingImages = emptyList()
                            imageError = null
                            scope.launch {
                                if (state.messages.isNotEmpty()) {
                                    listState.animateScrollToItem(state.messages.size - 1)
                                }
                            }
                        },
                        enabled = inputText.trim().isNotBlank() || pendingImages.isNotEmpty(),
                    ) {
                        Icon(Icons.Default.Send, contentDescription = "Send")
                    }
                }
            }
        }
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 12.dp),
            state = listState,
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            items(state.messages, key = { it.id }) { msg ->
                MessageRow(msg)
            }
            if (state.isAwaitingResponse && !state.messages.any { it.sender == Sender.AI && it.isStreaming }) {
                item(key = "incoming") { IncomingIndicator() }
            }
        }
    }
}

@Composable
private fun IncomingIndicator() {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Start,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = "> Incoming",
            fontFamily = FontFamily.Monospace,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
        )
    }
}

@Composable
private fun MessageRow(msg: ChatMessage) {
    val isUser = msg.sender == Sender.USER
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
    ) {
        Column(
            modifier = Modifier
                .clip(RoundedCornerShape(14.dp))
                .background(
                    if (isUser) MaterialTheme.colorScheme.surfaceVariant
                    else Color.Transparent
                )
                .padding(if (isUser) 10.dp else 0.dp)
                .fillMaxWidth(0.92f),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            if (isUser && msg.images.isNotEmpty()) {
                AttachmentsGrid(msg.images)
            }
            if (isUser) {
                if (msg.text.isNotBlank()) Text(msg.text)
            } else {
                Text(
                    text = buildString {
                        append("> ")
                        append(msg.text)
                        if (msg.isStreaming) append("|")
                    },
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.9f),
                )
            }
        }
    }
}

@Composable
private fun AttachmentsGrid(images: List<ImageAttachment>) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        val rows = images.chunked(2)
        for (row in rows) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                for (img in row) {
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(120.dp)
                            .clip(RoundedCornerShape(10.dp))
                            .background(MaterialTheme.colorScheme.surface)
                    ) {
                        Base64Image(
                            modifier = Modifier.fillMaxSize(),
                            mime = img.mime,
                            dataB64 = img.dataB64,
                        )
                    }
                }
                if (row.size == 1) {
                    Spacer(Modifier.weight(1f))
                }
            }
        }
    }
}

@Composable
private fun PendingImagesRow(images: List<PendingImage>, onRemove: (String) -> Unit) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        images.forEach { pending ->
            Box(
                modifier = Modifier
                    .size(56.dp)
                    .clip(RoundedCornerShape(10.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant),
            ) {
                pending.bitmap?.let { bmp ->
                    BitmapImage(
                        modifier = Modifier.fillMaxSize(),
                        bitmap = bmp,
                    )
                }
                IconButton(
                    modifier = Modifier.align(Alignment.TopEnd).size(24.dp),
                    onClick = { onRemove(pending.id) },
                ) {
                    Text("X")
                }
            }
        }
    }
}

data class PendingImage(
    val id: String,
    val attachment: ImageAttachment,
    val bitmap: android.graphics.Bitmap?,
)

data class PickedImagesResult(
    val images: List<PendingImage>,
    val error: String? = null,
)

suspend fun processPickedImages(
    context: android.content.Context,
    uris: List<Uri>,
    maxEdge: Int,
    maxMb: Int,
): PickedImagesResult {
    val out = mutableListOf<PendingImage>()
    var error: String? = null
    for (uri in uris) {
        val processed = processSingleImageUri(context, uri, maxEdge = maxEdge, maxMb = maxMb)
        if (processed == null) {
            error = "Failed to process one or more images."
            continue
        }
        out.add(processed)
        if (out.size >= 10) break
    }
    return PickedImagesResult(images = out, error = error)
}
