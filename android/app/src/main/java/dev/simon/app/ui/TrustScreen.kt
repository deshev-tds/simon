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
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import dev.simon.app.net.TlsCertificateInfo
import dev.simon.app.viewmodel.AppRoute
import dev.simon.app.viewmodel.TrustMode
import dev.simon.app.viewmodel.UiState

@Composable
fun TrustScreen(
    state: UiState,
    route: AppRoute.Trust,
    onTrust: () -> Unit,
    onCancel: () -> Unit,
) {
    val title = when (route.mode) {
        TrustMode.FIRST -> "Trust Certificate"
        TrustMode.CHANGED -> "Certificate Changed"
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(text = title, style = MaterialTheme.typography.headlineSmall)

        val routeError = route.error
        if (route.mode == TrustMode.CHANGED && !route.oldSha256Hex.isNullOrBlank()) {
            Text(
                text = "Previously trusted fingerprint:\n${route.oldSha256Hex}",
                style = MaterialTheme.typography.bodySmall,
            )
        }

        if (route.isLoading) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                CircularProgressIndicator()
                Text("Fetching server certificate...")
            }
        }

        if (!routeError.isNullOrBlank()) {
            Text(
                text = routeError,
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall,
            )
        }

        val cert = route.certificate
        if (cert != null) {
            CertificateDetails(cert)
        }

        Spacer(Modifier.height(8.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            OutlinedButton(
                modifier = Modifier.weight(1f),
                onClick = onCancel,
            ) {
                Text("Cancel")
            }
            Button(
                modifier = Modifier.weight(1f),
                enabled = cert != null && !route.isLoading,
                onClick = onTrust,
            ) {
                Text("Trust & Continue")
            }
        }

        val stateError = state.error
        if (!stateError.isNullOrBlank() && cert == null && !route.isLoading) {
            Text(
                text = stateError,
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

@Composable
private fun CertificateDetails(cert: TlsCertificateInfo) {
    val warn = if (cert.hostnameMatches) "" else " (hostname mismatch)"

    Text(
        text = "Server: ${cert.host}:${cert.port}$warn",
        style = MaterialTheme.typography.bodyMedium,
    )
    Text(
        text = "SHA-256 fingerprint:\n${cert.sha256FingerprintHex}",
        style = MaterialTheme.typography.bodySmall,
    )
    Text(
        text = "Subject:\n${cert.subject}",
        style = MaterialTheme.typography.bodySmall,
    )
    Text(
        text = "Issuer:\n${cert.issuer}",
        style = MaterialTheme.typography.bodySmall,
    )
    Text(
        text = "Serial: ${cert.serialHex}",
        style = MaterialTheme.typography.bodySmall,
    )
    Text(
        text = "Valid: ${cert.notBeforeIsoUtc} .. ${cert.notAfterIsoUtc}",
        style = MaterialTheme.typography.bodySmall,
    )
    if (!cert.hostnameMatches) {
        Text(
            text = "Warning: the certificate is not valid for the configured host. This is common when using an IP with a localhost certificate. Pinning will still enforce identity by certificate.",
            color = MaterialTheme.colorScheme.tertiary,
            style = MaterialTheme.typography.bodySmall,
        )
    }
}
