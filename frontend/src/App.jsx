import { useCallback, useEffect, useRef, useState } from "react";

const MODES = [
  { id: "full", label: "Full Page OCR", hint: "Extract all text from the document, streamed live." },
  { id: "spotting", label: "Text Spotting", hint: "Detect each text line and its position on the image." },
];

const ACCEPTED = ["image/jpeg", "image/png", "image/webp"];

async function readError(response) {
  try {
    const data = await response.json();
    return data.detail || `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
}

function formatNumber(val) {
  if (val === null || val === undefined) return "-";
  return Number(val).toLocaleString();
}

function parseDurationMs(str) {
  if (!str || typeof str !== "string") return 0;
  let total = 0;
  const h = str.match(/(\d+(?:\.\d+)?)h/);
  const m = str.match(/(\d+(?:\.\d+)?)m(?!s)/);
  const s = str.match(/(\d+(?:\.\d+)?)s/);
  const ms = str.match(/(\d+(?:\.\d+)?)ms/);
  if (h) total += parseFloat(h[1]) * 3600000;
  if (m) total += parseFloat(m[1]) * 60000;
  if (s) total += parseFloat(s[1]) * 1000;
  if (ms) total += parseFloat(ms[1]);
  return Math.round(total);
}

function formatCountdown(ms) {
  if (!ms || ms <= 0) return "Full quota";
  const totalSec = Math.ceil(ms / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  const parts = [];
  if (h > 0) parts.push(`${h}h`);
  if (m > 0 || h > 0) parts.push(`${m}m`);
  parts.push(`${s}s`);
  return `Refills to full in ${parts.join(" ")}`;
}

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [mode, setMode] = useState("full");
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [fullText, setFullText] = useState("");
  const [boxes, setBoxes] = useState([]);
  const [rawJson, setRawJson] = useState("");
  const [showRaw, setShowRaw] = useState(false);
  const [activeIndex, setActiveIndex] = useState(null);
  const [copied, setCopied] = useState(null);
  const [health, setHealth] = useState(null);
  const [usage, setUsage] = useState(null);
  const [sessionTokens, setSessionTokens] = useState(0);
  const [rateLimit, setRateLimit] = useState(null);
  const [now, setNow] = useState(() => Date.now());

  const inputRef = useRef(null);
  const abortRef = useRef(null);

  const applyRateLimit = useCallback((rl) => {
    if (!rl) return;
    const receivedAt = Date.now();
    setNow(receivedAt);
    setRateLimit({
      ...rl,
      receivedAt,
      resetRequestsMs: parseDurationMs(rl.reset_requests),
      resetTokensMs: parseDurationMs(rl.reset_tokens),
    });
  }, []);

  useEffect(() => {
    if (!rateLimit) return;
    const maxDuration = Math.max(rateLimit.resetRequestsMs || 0, rateLimit.resetTokensMs || 0);
    if (maxDuration <= 0) return;

    const timer = setInterval(() => {
      const current = Date.now();
      setNow(current);
      if (current - rateLimit.receivedAt >= maxDuration + 500) {
        clearInterval(timer);
      }
    }, 250);

    return () => clearInterval(timer);
  }, [rateLimit]);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "offline" }));
  }, []);

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const resetResults = () => {
    setFullText("");
    setBoxes([]);
    setRawJson("");
    setError("");
    setActiveIndex(null);
  };

  const applyUsage = (usageData) => {
    if (!usageData) return;
    setUsage(usageData);
    setSessionTokens((prev) => prev + (usageData.total_tokens || 0));
  };

  const selectFile = useCallback((selected) => {
    if (!selected) return;
    if (!ACCEPTED.includes(selected.type)) {
      setError("Please upload a JPG, PNG, or WEBP image.");
      return;
    }
    abortRef.current?.abort();
    setLoading(false);
    resetResults();
    setFile(selected);
  }, []);

  useEffect(() => {
    const onPaste = (e) => {
      const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
      if (item) selectFile(item.getAsFile());
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [selectFile]);

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    selectFile(e.dataTransfer.files?.[0]);
  };

  const run = async () => {
    if (!file || loading) return;
    resetResults();
    setLoading(true);

    const controller = new AbortController();
    abortRef.current = controller;
    const form = new FormData();
    form.append("file", file);

    try {
      const response = await fetch(`/api/ocr/${mode}`, {
        method: "POST",
        body: form,
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(await readError(response));

      if (mode === "full") {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        const handleLine = (line) => {
          const trimmed = line.trim();
          if (!trimmed) return;
          try {
            const event = JSON.parse(trimmed);
            if (event.type === "token") {
              setFullText((prev) => prev + (event.text || ""));
            } else if (event.type === "rate_limit") {
              applyRateLimit(event.data);
            } else if (event.type === "usage") {
              applyUsage(event.data);
            } else if (event.type === "error") {
              setError(event.detail || "Stream error");
            }
          } catch {
            setFullText((prev) => prev + line);
          }
        };

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";
          for (const line of lines) {
            handleLine(line);
          }
        }
        if (buffer.trim()) {
          handleLine(buffer);
        }
      } else {
        const data = await response.json();
        setBoxes(data.boxes || []);
        setRawJson(data.raw || "");
        if (data.rate_limit) applyRateLimit(data.rate_limit);
        if (data.usage) applyUsage(data.usage);
        if (!data.boxes?.length) setError("The model did not return any readable bounding boxes.");
      }
    } catch (err) {
      if (err.name !== "AbortError") setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  };

  const stop = () => abortRef.current?.abort();

  const copy = async (text, key) => {
    await navigator.clipboard.writeText(text);
    setCopied(key);
    setTimeout(() => setCopied(null), 1200);
  };

  const clearImage = () => {
    abortRef.current?.abort();
    setFile(null);
    setPreviewUrl(null);
    resetResults();
  };

  const hasResult = fullText || boxes.length > 0;
  const currentMode = MODES.find((m) => m.id === mode);

  const elapsedMs = rateLimit ? Math.max(0, now - rateLimit.receivedAt) : 0;

  const reqResetRemainingMs = rateLimit ? Math.max(0, (rateLimit.resetRequestsMs || 0) - elapsedMs) : 0;
  const liveRemainingRequests = (() => {
    if (!rateLimit?.limit_requests || rateLimit.remaining_requests === null) return null;
    if (reqResetRemainingMs <= 0) return rateLimit.limit_requests;
    const msPerRequest = 86400000 / rateLimit.limit_requests;
    const stillMissing = Math.ceil(reqResetRemainingMs / msPerRequest);
    return Math.min(
      rateLimit.limit_requests,
      Math.max(rateLimit.remaining_requests, rateLimit.limit_requests - stillMissing)
    );
  })();

  const tokResetRemainingMs = rateLimit ? Math.max(0, (rateLimit.resetTokensMs || 0) - elapsedMs) : 0;
  const liveRemainingTokens = (() => {
    if (!rateLimit?.limit_tokens || rateLimit.remaining_tokens === null) return null;
    if (tokResetRemainingMs <= 0 || !rateLimit.resetTokensMs) return rateLimit.limit_tokens;
    const missingTokens = rateLimit.limit_tokens - rateLimit.remaining_tokens;
    const fractionLeft = tokResetRemainingMs / rateLimit.resetTokensMs;
    return Math.min(
      rateLimit.limit_tokens,
      Math.floor(rateLimit.limit_tokens - missingTokens * fractionLeft)
    );
  })();

  const tokenRatio =
    rateLimit?.limit_tokens && liveRemainingTokens !== null
      ? Math.max(0, Math.min(100, (liveRemainingTokens / rateLimit.limit_tokens) * 100))
      : null;

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="logo">OCR</div>
          <div>
            <h1>Qwen Vision OCR</h1>
            <p>Extract and locate text in images with Qwen 3.8 on Groq</p>
          </div>
        </div>
        <div className={`status ${health?.api_key_configured ? "ok" : health ? "bad" : ""}`}>
          <span className="dot" />
          {!health && "Connecting..."}
          {health?.status === "offline" && "Backend offline"}
          {health?.status === "ok" && (health.api_key_configured ? health.model : "GROQ_API_KEY missing")}
        </div>
      </header>

      <section className="metrics-bar">
        <div className="metric-card">
          <span className="metric-label">Last Request Usage</span>
          <div className="metric-value">
            {usage ? `${formatNumber(usage.total_tokens)} tokens` : "-"}
          </div>
          <span className="metric-sub">
            {usage
              ? `In: ${formatNumber(usage.prompt_tokens)} / Out: ${formatNumber(usage.completion_tokens)} (${usage.total_time}s)`
              : "No requests yet"}
          </span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Session Total</span>
          <div className="metric-value">{formatNumber(sessionTokens)} tokens</div>
          <span className="metric-sub">Cumulative in current tab</span>
        </div>

        <div className="metric-card">
          <div className="metric-row">
            <span className="metric-label">Available Token Remaining</span>
            {tokenRatio !== null && <span className="metric-pct">{tokenRatio.toFixed(0)}%</span>}
          </div>
          <div className="metric-value">
            {rateLimit?.limit_tokens
              ? `${formatNumber(liveRemainingTokens)} / ${formatNumber(rateLimit.limit_tokens)}`
              : "-"}
          </div>
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: tokenRatio !== null ? `${tokenRatio}%` : "0%" }}
            />
          </div>
        </div>

        <div className="metric-card">
          <span className="metric-label">Available Request Remaining</span>
          <div className="metric-value">
            {rateLimit?.limit_requests
              ? `${formatNumber(liveRemainingRequests)} / ${formatNumber(rateLimit.limit_requests)}`
              : "-"}
          </div>
          <span className="metric-sub">
            {rateLimit ? formatCountdown(reqResetRemainingMs) : "Rolling 24h request quota"}
          </span>
        </div>
      </section>

      <div className="limit-banner">
        <strong>Free-Tier Token Limit Notice:</strong> This app runs on Groq's free tier for{" "}
        <code>qwen/qwen3.8-27b</code>, which restricts output to <strong>1,000 tokens per minute</strong> (~900
        tokens max per request). Because of this cap, it works best on images with a small-to-moderate amount of
        text (short receipts, signs, snippets, or cards). Dense documents with many lines may have their output
        truncated once the token limit is reached.
      </div>

      <main className="layout">
        <section className="panel image-panel">
          {!previewUrl ? (
            <div
              className={`dropzone ${dragging ? "dragging" : ""}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
            >
              <div className="drop-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 19V5M5 12l7-7 7 7" />
                </svg>
              </div>
              <h2>Drop an image here</h2>
              <p>or click to browse / paste with Ctrl+V</p>
              <span className="formats">JPG / PNG / WEBP</span>
            </div>
          ) : (
            <>
              <div className="image-toolbar">
                <span className="filename" title={file?.name}>{file?.name || "Pasted image"}</span>
                <div className="toolbar-actions">
                  <button className="ghost" onClick={() => inputRef.current?.click()}>Replace</button>
                  <button className="ghost" onClick={clearImage}>Remove</button>
                </div>
              </div>
              <div className="image-stage">
                <div className="image-wrap">
                  <img src={previewUrl} alt="Uploaded" />
                  {boxes.length > 0 && (
                    <svg className="overlay" viewBox="0 0 1 1" preserveAspectRatio="none">
                      {boxes.map((b, i) => (
                        <rect
                          key={i}
                          x={b.box[0]}
                          y={b.box[1]}
                          width={Math.max(b.box[2] - b.box[0], 0.001)}
                          height={Math.max(b.box[3] - b.box[1], 0.001)}
                          className={activeIndex === i ? "active" : ""}
                          vectorEffect="non-scaling-stroke"
                          onMouseEnter={() => setActiveIndex(i)}
                          onMouseLeave={() => setActiveIndex(null)}
                          onClick={() => copy(b.text, `box-${i}`)}
                        />
                      ))}
                    </svg>
                  )}
                  {loading && <div className="scanline" />}
                </div>
              </div>
            </>
          )}
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED.join(",")}
            hidden
            onChange={(e) => {
              selectFile(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
        </section>

        <section className="panel result-panel">
          <div className="tabs">
            {MODES.map((m) => (
              <button
                key={m.id}
                className={`tab ${mode === m.id ? "active" : ""}`}
                onClick={() => {
                  if (loading) return;
                  setMode(m.id);
                  resetResults();
                }}
                disabled={loading}
              >
                {m.label}
              </button>
            ))}
          </div>
          <p className="hint">{currentMode.hint}</p>

          <div className="actions">
            {loading ? (
              <button className="primary danger" onClick={stop}>Stop</button>
            ) : (
              <button className="primary" onClick={run} disabled={!file}>
                {mode === "full" ? "Extract Text" : "Spot Text"}
              </button>
            )}
            {loading && (
              <span className="spinner-label">
                <span className="spinner" /> {mode === "full" ? "Reading..." : "Locating text..."}
              </span>
            )}
          </div>

          {error && <div className="error">{error}</div>}

          <div className="result">
            {!hasResult && !loading && !error && (
              <div className="empty">
                {file ? "Ready. Choose a mode and run it." : "Upload an image to get started."}
              </div>
            )}

            {mode === "full" && (fullText || loading) && (
              <div className="text-result">
                <div className="result-header">
                  <span>Extracted text</span>
                  {fullText && !loading && (
                    <button className="ghost small" onClick={() => copy(fullText, "full")}>
                      {copied === "full" ? "Copied" : "Copy"}
                    </button>
                  )}
                </div>
                <pre className={loading ? "streaming" : ""}>{fullText}</pre>
              </div>
            )}

            {mode === "spotting" && boxes.length > 0 && (
              <div className="lines-result">
                <div className="result-header">
                  <span>{boxes.length} lines detected</span>
                  <div className="toolbar-actions">
                    <button className="ghost small" onClick={() => setShowRaw((v) => !v)}>
                      {showRaw ? "Hide JSON" : "Show JSON"}
                    </button>
                    <button className="ghost small" onClick={() => copy(boxes.map((b) => b.text).join("\n"), "all")}>
                      {copied === "all" ? "Copied" : "Copy all"}
                    </button>
                  </div>
                </div>
                {showRaw ? (
                  <pre className="raw">{rawJson}</pre>
                ) : (
                  <ul className="lines">
                    {boxes.map((b, i) => (
                      <li
                        key={i}
                        className={activeIndex === i ? "active" : ""}
                        onMouseEnter={() => setActiveIndex(i)}
                        onMouseLeave={() => setActiveIndex(null)}
                        onClick={() => copy(b.text, `box-${i}`)}
                      >
                        <span className="index">{i + 1}</span>
                        <span className="line-text">{b.text || <em>(empty)</em>}</span>
                        <span className="copy-tag">{copied === `box-${i}` ? "Copied" : "Copy"}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
