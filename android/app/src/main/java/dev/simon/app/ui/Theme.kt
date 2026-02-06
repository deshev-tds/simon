package dev.simon.app.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val DarkColors = darkColorScheme(
    primary = Color(0xFF7EE787),
    secondary = Color(0xFF58A6FF),
    tertiary = Color(0xFFE3B341),
    background = Color(0xFF0D1117),
    surface = Color(0xFF0D1117),
    onPrimary = Color(0xFF0D1117),
    onSecondary = Color(0xFF0D1117),
    onBackground = Color(0xFFC9D1D9),
    onSurface = Color(0xFFC9D1D9),
)

private val LightColors = lightColorScheme(
    primary = Color(0xFF1F6FEB),
    secondary = Color(0xFF0969DA),
    background = Color(0xFFFFFFFF),
    surface = Color(0xFFFFFFFF),
    onPrimary = Color(0xFFFFFFFF),
    onSecondary = Color(0xFFFFFFFF),
    onBackground = Color(0xFF0D1117),
    onSurface = Color(0xFF0D1117),
)

@Composable
fun SimonTheme(
    content: @Composable () -> Unit,
) {
    val dark = isSystemInDarkTheme()
    MaterialTheme(
        colorScheme = if (dark) DarkColors else LightColors,
        typography = MaterialTheme.typography,
        content = content,
    )
}

