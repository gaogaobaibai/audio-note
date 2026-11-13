// Enrolment by microphone or file: POST /api/speakers/{name}/samples?filename=&force= with the raw audio as body.
// The browser recording uses MediaRecorder with a level meter; shorter than 3 s is refused, 90 s is the cap.
// A voice that does not resemble the person's existing samples is answered with a 「仍然加入」 notice.
import { html } from "../../core/html.js";
import { render } from "../../core/dom.js";
import { api, isMismatch } from "../../core/api.js";
import { emit } from "../../core/bus.js";
import { fmtClock } from "../../core/format.js";
import { icon } from "../../ui/icons.js";
import { notice, spinner } from "../../ui/components.js";
import { mismatchNotice } from "../../ui/speaker.js";
import { openModal } from "../../ui/modal.js";
import { toast } from "../../ui/toast.js";

export function openEnroll(name = "", names = []) {
  let rec = null;      // {stop(abort)} while the browser is recording
  return openModal({
    title: name ? `为 ${name} 补充样本` : "补充声纹样本", size: "lg", actions: [{ label: "完成", kind: "primary" }],
    body: html`<p class="help">通常不需要：给「未识别的声音」起名就会自动取样本；这里是识别不准时的补充手段。</p>
      <div class="field"><label for="en-name">姓名</label><input class="input" id="en-name" list="en-list" value="${name}" placeholder="例如：张三" autocomplete="off" ${name ? "" : html`data-autofocus`}>
        <datalist id="en-list">${names.map((n) => html`<option value="${n}"></option>`)}</datalist></div>
      <div class="enroll-grid">
        <section class="enroll-way"><h3>${icon("mic", 16)}用麦克风录一段</h3>
          <div class="row nowrap"><button type="button" class="btn rec" id="rec-btn">${icon("mic", 16)}<span>开始录音</span></button><span class="rec-timer" id="rec-timer" aria-live="off">0:00</span><div class="level" aria-hidden="true"><i id="rec-level"></i></div></div>
          <p class="help">用平时说话的音量自然地说 15–30 秒（念一段文字就行），然后点停止。浏览器第一次会询问麦克风权限。</p>
          <div id="rec-result" aria-live="polite"></div></section>
        <section class="enroll-way"><h3>${icon("upload", 16)}或上传录音文件</h3>
          <input type="file" id="en-file" class="file-input" accept="audio/*,.m4a,.webm,.amr" multiple aria-label="选择录音文件">
          <p class="help">wav / mp3 / m4a / 微信语音等都可以；尽量只有这个人说话的片段，10–60 秒。可以一次选多个。</p>
          <div id="file-result" aria-live="polite"></div></section>
      </div>`,
    onMount: (h) => {
      const $ = (sel) => h.body.querySelector(sel);
      const getName = () => { const n = $("#en-name").value.trim(); if (!n) { toast("先填姓名", { kind: "err" }); $("#en-name").focus(); } return n; };
      // returns "ok" | "mismatch" | "error"
      const upload = async (blob, filename, where, { force = false } = {}) => {
        const n = getName();
        if (!n) return "error";
        render(where, spinner("正在提取声纹…"));
        try {
          const r = await api(`/api/speakers/${encodeURIComponent(n)}/samples`, { method: "POST", query: { filename, force: force ? "true" : null }, body: blob });
          if (!h.closed) render(where, notice({ kind: "ok", compact: true, body: `已登记 ${n}：这段有 ${r.speech_s} 秒语音，现在共 ${r.samples} 个样本。` }));
          toast(`已登记 ${n}`, { kind: "ok" });
          emit("speakers:changed", { reason: "enroll" });
          return "ok";
        } catch (e) {
          if (h.closed) return "error";
          if (isMismatch(e)) {
            toast(e.message, { kind: "err", ms: 8000 });
            render(where, mismatchNotice(e.data, "data-enforce"));
            where.querySelector("[data-enforce]").addEventListener("click", () => upload(blob, filename, where, { force: true }));
            return "mismatch";
          }
          render(where, notice({ kind: "danger", compact: true, body: e.message || String(e) }));
          return "error";
        }
      };

      $("#en-file").addEventListener("change", async (e) => {
        const files = Array.from(e.target.files);
        e.target.value = "";
        if (!files.length || !getName()) return;
        const box = $("#file-result");
        // one result box per file; a mismatch on one file does not stop the rest of the batch
        render(box, html`${files.map((f, i) => html`<div class="file-res" data-fi="${i}"><div class="small t2 clip">${f.name}</div><div data-body><span class="t3 small">等待上传…</span></div></div>`)}`);
        for (let i = 0; i < files.length; i++) {
          const body = box.querySelector(`[data-fi="${i}"] [data-body]`);
          if (!body || h.closed) break;
          await upload(files[i], files[i].name, body);
        }
      });

      const btn = $("#rec-btn");
      const setBtn = (recording) => { render(btn, recording ? html`${icon("stop", 14)}<span>停止并登记</span>` : html`${icon("mic", 16)}<span>开始录音</span>`); btn.classList.toggle("rec", !recording); };
      btn.addEventListener("click", async () => {
        if (rec) { rec.stop(false); return; }
        if (!getName()) return;
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) { toast("这个浏览器不支持录音，请用上传文件", { kind: "err" }); return; }
        let stream;
        try { stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false } }); }
        catch (e) { toast(`无法打开麦克风：${e.message}`, { kind: "err", ms: 6000 }); return; }
        if (h.closed) { stream.getTracks().forEach((t) => t.stop()); return; }
        const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"].find((m) => MediaRecorder.isTypeSupported(m)) || "";
        const mr = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
        const chunks = [];
        mr.ondataavailable = (ev) => { if (ev.data.size) chunks.push(ev.data); };
        const ac = new (window.AudioContext || window.webkitAudioContext)();
        const an = ac.createAnalyser();
        an.fftSize = 512;
        ac.createMediaStreamSource(stream).connect(an);
        const buf = new Uint8Array(an.frequencyBinCount);
        const t0 = Date.now();
        const timer = setInterval(() => {
          an.getByteTimeDomainData(buf);
          let peak = 0;
          for (const v of buf) peak = Math.max(peak, Math.abs(v - 128) / 128);
          const lv = $("#rec-level");
          if (lv) lv.style.width = `${Math.min(100, peak * 160)}%`;
          const s = Math.floor((Date.now() - t0) / 1000);
          const tm = $("#rec-timer");
          if (tm) tm.textContent = fmtClock(s);
          if (s >= 90 && rec) rec.stop(false);
        }, 100);
        setBtn(true);
        rec = {
          stop: (abort) => {
            clearInterval(timer);
            try { ac.close(); } catch (e) { /* ignore */ }
            mr.onstop = async () => {
              stream.getTracks().forEach((t) => t.stop());
              rec = null;
              if (!h.closed) { setBtn(false); const lv = $("#rec-level"); if (lv) lv.style.width = "0"; }
              if (abort || h.closed) return;
              if ((Date.now() - t0) / 1000 < 3) { toast("录得太短了，至少 5 秒", { kind: "err" }); return; }
              const type = mr.mimeType || "audio/webm";
              const ext = type.includes("ogg") ? "ogg" : type.includes("mp4") ? "mp4" : "webm";
              await upload(new Blob(chunks, { type }), `record.${ext}`, $("#rec-result"));
            };
            if (mr.state !== "inactive") mr.stop(); else mr.onstop();
          },
        };
        mr.start(250);
      });
      h.stopRecording = () => { if (rec) { try { rec.stop(true); } catch (e) { /* ignore */ } rec = null; } };
    },
    onClose: () => { if (rec) { try { rec.stop(true); } catch (e) { /* ignore */ } rec = null; } },
  });
}
