package dev.simon.app.ui

import androidx.compose.material3.DrawerValue
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import dev.simon.app.model.ImageAttachment
import dev.simon.app.viewmodel.UiState
import kotlinx.coroutines.launch

@Composable
fun MainScreen(
    state: UiState,
    onSend: (text: String, images: List<ImageAttachment>) -> Unit,
    onRefreshSessions: () -> Unit,
    onSelectSession: (id: Long) -> Unit,
    onNewSession: () -> Unit,
    onStartRecording: () -> Unit,
    onStopRecording: () -> Unit,
    onInterrupt: () -> Unit,
    onOpenSetup: () -> Unit,
) {
    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    var voiceOpen by remember { mutableStateOf(false) }

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            SessionDrawerContent(
                sessions = state.sessions,
                currentSessionId = state.currentSessionId,
                isLoading = state.isLoadingSession,
                onSelectSession = { id ->
                    onSelectSession(id)
                    scope.launch { drawerState.close() }
                },
                onNewSession = {
                    onNewSession()
                    scope.launch { drawerState.close() }
                },
                onOpenSetup = {
                    onOpenSetup()
                    scope.launch { drawerState.close() }
                },
                onRefresh = onRefreshSessions,
            )
        }
    ) {
        ChatScreen(
            state = state,
            onOpenDrawer = { scope.launch { drawerState.open() } },
            onSend = onSend,
            onOpenVoice = { voiceOpen = true },
        )

        if (voiceOpen) {
            VoiceOverlay(
                state = state,
                onClose = {
                    if (state.isRecording) onStopRecording()
                    else if (state.aiIsSpeaking || state.isProcessing) onInterrupt()
                    voiceOpen = false
                },
                onStartRecording = onStartRecording,
                onStopRecording = onStopRecording,
                onInterrupt = onInterrupt,
            )
        }
    }
}

