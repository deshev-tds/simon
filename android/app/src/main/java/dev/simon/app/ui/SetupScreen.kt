package dev.simon.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import dev.simon.app.viewmodel.UiState

@Composable
fun SetupScreen(
    state: UiState,
    onSave: (host: String, port: Int) -> Unit,
    onForgetTrust: () -> Unit,
) {
    var host by remember { mutableStateOf("") }
    var portText by remember { mutableStateOf("8000") }
    val errorText = state.error

    LaunchedEffect(state.serverHost, state.serverPort) {
        host = state.serverHost.orEmpty()
        portText = state.serverPort.toString()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            text = "Simon Server",
            style = MaterialTheme.typography.headlineSmall,
        )
        Text(
            text = "TLS only. The app will connect via HTTPS/WSS and ask you to trust the server certificate.",
            style = MaterialTheme.typography.bodyMedium,
        )

        OutlinedTextField(
            modifier = Modifier.fillMaxWidth(),
            value = host,
            onValueChange = { host = it },
            label = { Text("Host (IP or DNS)") },
            singleLine = true,
        )

        OutlinedTextField(
            modifier = Modifier.fillMaxWidth(),
            value = portText,
            onValueChange = { portText = it.filter { ch -> ch.isDigit() }.take(5) },
            label = { Text("Port") },
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            singleLine = true,
        )

        if (!errorText.isNullOrBlank()) {
            Text(
                text = errorText,
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall,
            )
        }

        if (!state.pinnedCertSha256Hex.isNullOrBlank()) {
            Text(
                text = "Pinned certificate fingerprint:\n${state.pinnedCertSha256Hex}",
                style = MaterialTheme.typography.bodySmall,
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Button(onClick = onForgetTrust) {
                    Text("Forget Trust")
                }
            }
        }

        Spacer(Modifier.height(8.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.End,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Button(
                enabled = host.trim().isNotEmpty() && portText.toIntOrNull() != null,
                onClick = {
                    val port = portText.toIntOrNull() ?: 8000
                    onSave(host.trim(), port)
                }
            ) {
                Text("Continue")
            }
        }
    }
}
