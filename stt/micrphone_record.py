import numpy as np
from flask import Flask, render_template_string
from flask_sock import Sock
from faster_whisper import WhisperModel

app = Flask(__name__)
sock = Sock(app)

print("Loading Whisper model...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("Model loaded.")

SAMPLE_RATE = 16000
CHUNK_SECONDS = 3          # how much audio to accumulate before transcribing
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>Real-Time Transcription</title>
  <style>
    body { font-family: sans-serif; max-width: 700px; margin: 40px auto; }
    #transcript { border: 1px solid #ccc; padding: 15px; min-height: 200px; white-space: pre-wrap; }
    button { padding: 10px 20px; font-size: 16px; margin-bottom: 15px; }
    #status { color: gray; margin-bottom: 10px; }
  </style>
</head>
<body>
  <h2>Real-Time Speech Transcription (faster-whisper)</h2>
  <button id="toggleBtn">Start Recording</button>
  <div id="status">Idle</div>
  <div id="transcript"></div>

  <script>
    let audioContext, processor, source, socket, stream;
    let recording = false;

    const btn = document.getElementById('toggleBtn');
    const statusEl = document.getElementById('status');
    const transcriptEl = document.getElementById('transcript');

    btn.onclick = async () => {
      if (!recording) await startRecording();
      else stopRecording();
    };

    function floatTo16BitPCM(float32Array) {
      const buffer = new Int16Array(float32Array.length);
      for (let i = 0; i < float32Array.length; i++) {
        let s = Math.max(-1, Math.min(1, float32Array[i]));
        buffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      return buffer;
    }

    // Simple linear-interpolation resampler
    function downsampleBuffer(buffer, inputRate, outputRate) {
      if (outputRate === inputRate) return buffer;
      const ratio = inputRate / outputRate;
      const newLength = Math.round(buffer.length / ratio);
      const result = new Float32Array(newLength);
      let offsetResult = 0, offsetBuffer = 0;
      while (offsetResult < newLength) {
        const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
        let accum = 0, count = 0;
        for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
          accum += buffer[i];
          count++;
        }
        result[offsetResult] = accum / (count || 1);
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
      }
      return result;
    }

    async function startRecording() {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContext = new AudioContext(); // native sample rate, usually 44100/48000

      socket = new WebSocket(`ws://${location.host}/ws`);
      socket.binaryType = "arraybuffer";

      socket.onmessage = (event) => {
        transcriptEl.textContent += event.data + " ";
      };

      socket.onopen = () => {
        source = audioContext.createMediaStreamSource(stream);
        processor = audioContext.createScriptProcessor(4096, 1, 1);

        processor.onaudioprocess = (e) => {
          if (socket.readyState !== WebSocket.OPEN) return;
          const input = e.inputBuffer.getChannelData(0);
          const downsampled = downsampleBuffer(input, audioContext.sampleRate, 16000);
          const pcm16 = floatTo16BitPCM(downsampled);
          socket.send(pcm16.buffer);
        };

        source.connect(processor);
        processor.connect(audioContext.destination);

        recording = true;
        btn.textContent = "Stop Recording";
        statusEl.textContent = "Recording...";
      };
    }

    function stopRecording() {
      if (processor) processor.disconnect();
      if (source) source.disconnect();
      if (audioContext) audioContext.close();
      if (stream) stream.getTracks().forEach(t => t.stop());
      if (socket) socket.close();
      recording = false;
      btn.textContent = "Start Recording";
      statusEl.textContent = "Idle";
    }
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@sock.route("/ws")
def transcribe_ws(ws):
    buffer = np.zeros((0,), dtype=np.float32)

    while True:
        data = ws.receive()
        if data is None:
            break

        # Incoming data is raw Int16 PCM at 16kHz mono
        int16_chunk = np.frombuffer(data, dtype=np.int16)
        float_chunk = int16_chunk.astype(np.float32) / 32768.0
        buffer = np.concatenate([buffer, float_chunk])

        if len(buffer) >= CHUNK_SAMPLES:
            audio_segment = buffer[:CHUNK_SAMPLES]
            buffer = buffer[CHUNK_SAMPLES:]  # keep leftover for next round

            try:
                segments, _ = model.transcribe(
                    audio_segment, language="en", beam_size=1, vad_filter=True
                )
                text = " ".join(seg.text.strip() for seg in segments).strip()
                if text:
                    ws.send(text)
            except Exception as e:
                print("Transcription error:", e)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)