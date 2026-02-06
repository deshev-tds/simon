package dev.simon.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import dev.simon.app.model.SessionSummary
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun SessionDrawerContent(
    sessions: List<SessionSummary>,
    currentSessionId: Long?,
    isLoading: Boolean,
    onSelectSession: (Long) -> Unit,
    onNewSession: () -> Unit,
    onOpenSetup: () -> Unit,
    onRefresh: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxHeight()
            .width(320.dp)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Sessions", style = MaterialTheme.typography.headlineSmall)

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = onNewSession) { Text("New Chat") }
            OutlinedButton(onClick = onRefresh) { Text("Refresh") }
        }

        LazyColumn(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            items(sessions, key = { it.id }) { session ->
                val active = currentSessionId == session.id
                SessionRow(
                    session = session,
                    isActive = active,
                    onClick = { onSelectSession(session.id) },
                )
            }
        }

        if (isLoading) {
            Text("Loading...", style = MaterialTheme.typography.bodySmall)
        }

        Spacer(Modifier.height(4.dp))
        OutlinedButton(onClick = onOpenSetup) {
            Text("Server Settings")
        }
    }
}

@Composable
private fun SessionRow(session: SessionSummary, isActive: Boolean, onClick: () -> Unit) {
    val bg = if (isActive) MaterialTheme.colorScheme.primary.copy(alpha = 0.15f) else MaterialTheme.colorScheme.surface
    val title = session.title.trim().ifBlank { "Session ${session.id}" }
    val summary = session.summary?.trim().orEmpty()
    val dt = formatDate(session.updatedAtS.takeIf { it > 0.0 } ?: session.createdAtS)

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(bg)
            .clickable(onClick = onClick)
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text(title, style = MaterialTheme.typography.titleSmall)
        if (summary.isNotBlank()) {
            Text(summary, style = MaterialTheme.typography.bodySmall, maxLines = 1)
        }
        Text(dt, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
    }
}

private fun formatDate(tsSeconds: Double): String {
    if (tsSeconds <= 0.0) return ""
    val d = Date((tsSeconds * 1000.0).toLong())
    val fmt = SimpleDateFormat("yyyy-MM-dd", Locale.US)
    return fmt.format(d)
}
