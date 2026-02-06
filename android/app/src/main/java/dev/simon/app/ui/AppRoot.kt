package dev.simon.app.ui

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.simon.app.viewmodel.AppRoute
import dev.simon.app.viewmodel.MainViewModel

@Composable
fun AppRoot(viewModel: MainViewModel) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    when (val route = state.route) {
        is AppRoute.Setup -> SetupScreen(
            state = state,
            onSave = { host, port -> viewModel.saveServer(host, port) },
            onForgetTrust = { viewModel.forgetPinnedCertificate() },
        )
        is AppRoute.Trust -> TrustScreen(
            state = state,
            route = route,
            onTrust = { viewModel.trustDisplayedCertificate() },
            onCancel = { viewModel.backToSetup() },
        )
        is AppRoute.Main -> MainScreen(
            state = state,
            onSend = { text, images -> viewModel.sendMessage(text, images) },
            onRefreshSessions = { viewModel.refreshSessions() },
            onSelectSession = { id -> viewModel.switchSession(id) },
            onNewSession = { viewModel.createNewSession() },
            onStartRecording = { viewModel.startRecording() },
            onStopRecording = { viewModel.stopRecordingAndCommit() },
            onInterrupt = { viewModel.interrupt() },
            onOpenSetup = { viewModel.backToSetup() },
        )
    }
}
