package dev.simon.app.net

import android.util.Base64
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import java.security.MessageDigest
import java.security.SecureRandom
import java.security.cert.CertificateException
import java.security.cert.X509Certificate
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import javax.net.ssl.HostnameVerifier
import javax.net.ssl.HttpsURLConnection
import javax.net.ssl.SSLContext
import javax.net.ssl.SSLSocket
import javax.net.ssl.TrustManager
import javax.net.ssl.X509TrustManager

data class TlsCertificateInfo(
    val host: String,
    val port: Int,
    val derBase64: String,
    val sha256FingerprintHex: String,
    val subject: String,
    val issuer: String,
    val serialHex: String,
    val notBeforeIsoUtc: String,
    val notAfterIsoUtc: String,
    val hostnameMatches: Boolean,
)

class PinnedCertificateMismatchException(message: String) : CertificateException(message)

private fun sha256Hex(bytes: ByteArray): String {
    val digest = MessageDigest.getInstance("SHA-256").digest(bytes)
    val sb = StringBuilder(digest.size * 2)
    for (b in digest) sb.append(String.format("%02X", b))
    return sb.toString()
}

private fun formatIsoUtc(date: Date): String {
    val fmt = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US)
    fmt.timeZone = TimeZone.getTimeZone("UTC")
    return fmt.format(date)
}

private object InsecureTrustManager : X509TrustManager {
    override fun checkClientTrusted(chain: Array<X509Certificate>, authType: String) = Unit
    override fun checkServerTrusted(chain: Array<X509Certificate>, authType: String) = Unit
    override fun getAcceptedIssuers(): Array<X509Certificate> = emptyArray()
}

private class PinnedLeafTrustManager(private val pinnedLeafDer: ByteArray) : X509TrustManager {
    override fun checkClientTrusted(chain: Array<X509Certificate>, authType: String) {
        throw CertificateException("Client certs not supported")
    }

    override fun checkServerTrusted(chain: Array<X509Certificate>, authType: String) {
        if (chain.isEmpty()) throw CertificateException("Empty server certificate chain")
        val leaf = chain[0]
        val leafDer = leaf.encoded
        if (!leafDer.contentEquals(pinnedLeafDer)) {
            throw PinnedCertificateMismatchException("Pinned certificate mismatch")
        }
    }

    override fun getAcceptedIssuers(): Array<X509Certificate> = emptyArray()
}

private fun buildSslContext(trustManager: X509TrustManager): SSLContext {
    val sslContext = SSLContext.getInstance("TLS")
    sslContext.init(null, arrayOf<TrustManager>(trustManager), SecureRandom())
    return sslContext
}

suspend fun probeServerCertificate(host: String, port: Int): TlsCertificateInfo = withContext(Dispatchers.IO) {
    val sslContext = buildSslContext(InsecureTrustManager)
    val socket = (sslContext.socketFactory.createSocket(host, port) as SSLSocket)
    socket.soTimeout = 10_000
    socket.use {
        it.startHandshake()
        val session = it.session
        val cert = session.peerCertificates.firstOrNull() as? X509Certificate
            ?: throw CertificateException("No peer certificate")

        val der = cert.encoded
        val derB64 = Base64.encodeToString(der, Base64.NO_WRAP)
        val fp = sha256Hex(der)

        val hostnameMatches = HttpsURLConnection.getDefaultHostnameVerifier().verify(host, session)

        TlsCertificateInfo(
            host = host,
            port = port,
            derBase64 = derB64,
            sha256FingerprintHex = fp,
            subject = cert.subjectX500Principal?.name ?: "",
            issuer = cert.issuerX500Principal?.name ?: "",
            serialHex = cert.serialNumber?.toString(16)?.uppercase(Locale.US) ?: "",
            notBeforeIsoUtc = formatIsoUtc(cert.notBefore),
            notAfterIsoUtc = formatIsoUtc(cert.notAfter),
            hostnameMatches = hostnameMatches,
        )
    }
}

fun buildPinnedOkHttpClient(expectedHost: String, pinnedDerBase64: String): OkHttpClient {
    val pinnedDer = Base64.decode(pinnedDerBase64, Base64.DEFAULT)
    val trustManager = PinnedLeafTrustManager(pinnedDer)
    val sslContext = buildSslContext(trustManager)

    val hostnameVerifier = HostnameVerifier { hostname, _ ->
        // We rely on leaf pinning, but still ensure we only talk to the configured host.
        hostname.equals(expectedHost, ignoreCase = true)
    }

    return OkHttpClient.Builder()
        .sslSocketFactory(sslContext.socketFactory, trustManager)
        .hostnameVerifier(hostnameVerifier)
        .build()
}
