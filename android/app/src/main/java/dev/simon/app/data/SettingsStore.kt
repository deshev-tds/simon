package dev.simon.app.data

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "simon_settings")

data class AppSettings(
    val serverHost: String?,
    val serverPort: Int,
    val pinnedCertDerBase64: String?,
    val pinnedCertSha256Hex: String?,
    val lastSessionId: Long?,
)

class SettingsStore(private val context: Context) {
    private object Keys {
        val SERVER_HOST = stringPreferencesKey("server_host")
        val SERVER_PORT = intPreferencesKey("server_port")
        val PINNED_CERT_DER_B64 = stringPreferencesKey("pinned_cert_der_b64")
        val PINNED_CERT_SHA256_HEX = stringPreferencesKey("pinned_cert_sha256_hex")
        val LAST_SESSION_ID = longPreferencesKey("last_session_id")
    }

    val settingsFlow: Flow<AppSettings> = context.dataStore.data.map { prefs ->
        AppSettings(
            serverHost = prefs[Keys.SERVER_HOST],
            serverPort = prefs[Keys.SERVER_PORT] ?: 8000,
            pinnedCertDerBase64 = prefs[Keys.PINNED_CERT_DER_B64],
            pinnedCertSha256Hex = prefs[Keys.PINNED_CERT_SHA256_HEX],
            lastSessionId = prefs[Keys.LAST_SESSION_ID],
        )
    }

    suspend fun setServer(host: String, port: Int) {
        context.dataStore.edit { prefs ->
            prefs[Keys.SERVER_HOST] = host.trim()
            prefs[Keys.SERVER_PORT] = port
        }
    }

    suspend fun setPinnedCertificate(derBase64: String, sha256Hex: String) {
        context.dataStore.edit { prefs ->
            prefs[Keys.PINNED_CERT_DER_B64] = derBase64
            prefs[Keys.PINNED_CERT_SHA256_HEX] = sha256Hex
        }
    }

    suspend fun clearPinnedCertificate() {
        context.dataStore.edit { prefs ->
            prefs.remove(Keys.PINNED_CERT_DER_B64)
            prefs.remove(Keys.PINNED_CERT_SHA256_HEX)
        }
    }

    suspend fun setLastSessionId(id: Long?) {
        context.dataStore.edit { prefs ->
            if (id == null) prefs.remove(Keys.LAST_SESSION_ID)
            else prefs[Keys.LAST_SESSION_ID] = id
        }
    }
}

