//! Acoustic model: run the ONNX graph per chunk and stitch the utterance the way the reference does
//! (start-pause trim, end-token frames cut, 5 ms fades, 0.15 s between chunks).
use std::path::Path;

use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;

pub const SAMPLE_RATE: usize = 24000;
pub const GAP_SAMPLES: usize = 3600; // 0.15 s

pub struct Synth {
    _env: Environment,
    session: Session,
    pub voices: Vec<(String, Vec<f32>, usize)>,
}

/// numpy's round(): half to even.
fn round_half_even(x: f64) -> i64 {
    let f = x.floor();
    let d = x - f;
    if d > 0.5 {
        (f + 1.0) as i64
    } else if d < 0.5 {
        f as i64
    } else if (f as i64) % 2 == 0 {
        f as i64
    } else {
        (f + 1.0) as i64
    }
}

impl Synth {
    pub fn new(model: impl AsRef<Path>, voices: Vec<(String, Vec<f32>, usize)>, threads: usize) -> anyhow::Result<Synth> {
        let env = ort::init().build().map_err(|e| anyhow::anyhow!("onnxruntime init: {e}"))?;
        let mut builder = Session::builder(&env).map_err(|e| anyhow::anyhow!("session: {e}"))?;
        if threads > 0 {
            builder = builder.with_intra_threads(threads).map_err(|e| anyhow::anyhow!("threads: {e}"))?;
        }
        let session = builder
            .commit_from_file(model)
            .map_err(|e| anyhow::anyhow!("loading model: {e}"))?;
        Ok(Synth { _env: env, session, voices })
    }

    fn run(&mut self, tokens: &[i64], style: &[f32], speed: f32) -> anyhow::Result<(Vec<f32>, Vec<i64>)> {
        let ids = Tensor::from_array(([1usize, tokens.len()], tokens.to_vec()))?;
        let sty = Tensor::from_array(([1usize, style.len()], style.to_vec()))?;
        let spd = Tensor::from_array(([1usize], vec![speed]))?;
        let out = self
            .session
            .run(ort::inputs!("input_ids" => ids, "style" => sty, "speed" => spd))
            .map_err(|e| anyhow::anyhow!("run: {e}"))?;
        let (_, wav) = out["waveform"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("waveform: {e}"))?;
        let wav = wav.to_vec();
        let (_, dur) = out["durations"]
            .try_extract_tensor::<i64>()
            .map_err(|e| anyhow::anyhow!("durations: {e}"))?;
        let dur = dur.to_vec();
        Ok((wav, dur))
    }

    /// style row for a token sequence: one row per input length.
    pub fn style_for(&self, voice: &str, len: usize) -> Vec<f32> {
        let (_, data, cols) = self
            .voices
            .iter()
            .find(|(k, _, _)| k == voice)
            .unwrap_or_else(|| panic!("no voice {voice}"));
        let cols = *cols;
        let row = (len.min(data.len() / cols.max(1))).saturating_sub(1);
        data[row * cols..(row + 1) * cols].to_vec()
    }

    fn chunk(&mut self, tokens: &[i64], style: &[f32], speed: f32, keep_start: bool) -> Vec<f32> {
        let (wav, dur) = self.run(tokens, style, speed).expect("onnx run failed");
        let w: Vec<f32> = wav.iter().map(|v| v.clamp(-1.0, 1.0)).collect();
        let total: i64 = dur.iter().sum();
        let len = w.len();
        let spf = round_half_even(len as f64 / (total.max(1)) as f64).max(1) as usize;
        let lead = if keep_start { 0 } else { (dur[0].max(1) - 1) as usize * spf };
        let trail = (dur[dur.len() - 1].max(1) - 1) as usize * spf;
        let mut w = w;
        if lead + trail + spf < len {
            let end = if trail > 0 { len - trail } else { len };
            let start = lead.min(end);
            w = w[start..end].to_vec();
        }
        let f = (120usize).min(w.len() / 8);
        if f > 1 {
            for i in 0..f {
                w[i] *= i as f32 / (f - 1) as f32;
            }
            for i in 0..f {
                let idx = w.len() - 1 - i;
                w[idx] *= i as f32 / (f - 1) as f32;
            }
        }
        w
    }

    pub fn synthesize(&mut self, chunks: &[Vec<i64>], voice: &str, speed: f32) -> Vec<f32> {
        let mut out: Vec<f32> = Vec::new();
        for (i, tok) in chunks.iter().enumerate() {
            let style = self.style_for(voice, tok.len());
            let w = self.chunk(tok, &style, speed, i == 0);
            if i > 0 {
                out.extend_from_slice(&[0.0f32; GAP_SAMPLES]);
            }
            out.extend_from_slice(&w);
        }
        out
    }
}

pub mod wav {
    /// Write 16-bit PCM mono; libsndfile's scaling is x*32768 clamped to i16.
    pub fn write(path: &str, samples: &[f32], rate: usize) -> std::io::Result<()> {
        let data_len = samples.len() * 2;
        let mut buf: Vec<u8> = Vec::with_capacity(44 + data_len);
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
        buf.extend_from_slice(b"WAVEfmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&1u16.to_le_bytes()); // mono
        buf.extend_from_slice(&(rate as u32).to_le_bytes());
        buf.extend_from_slice(&(rate as u32 * 2).to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&(data_len as u32).to_le_bytes());
        for s in samples {
            let v = (s.clamp(-1.0, 1.0) * 32768.0).round();
            let v = v.clamp(-32768.0, 32767.0) as i16;
            buf.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, buf)
    }
}
