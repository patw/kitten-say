//! The char-level BART G2P (KittenML/g2p_bart_en_us, 0.75M params, 1 encoder + 1 decoder layer,
//! d_model 128, 1 head) implemented in plain Rust: greedy decode, tied output embedding.
use std::collections::HashMap;

use crate::assets::gunzip;
use crate::assets::BART_GZ;

pub const D_MODEL: usize = 128;
pub const VOCAB: usize = 63;
pub const FFN: usize = 1024;
pub const MAX_LEN: usize = 64;
pub const BOS: usize = 1;
pub const EOS: usize = 2;

pub const GRAPHEMES: &str = "____AIOWYbdfhijklmnpstuvwz'-.BCDEFGHJKLMNPQRSTUVXZacegoqrxy";
pub const PHONEMES: &str = "____AIOWYbdfhijklmnpstuvwz\u{e6}\u{f0}\u{14b}\u{251}\u{254}\u{259}\u{25b}\u{25c}\u{261}\u{26a}\u{279}\u{27e}\u{283}\u{28a}\u{28c}\u{292}\u{294}\u{2a4}\u{2a7}\u{2c8}\u{2cc}\u{3b8}\u{1d4a}\u{1d7b}";
fn gelu(x: f32) -> f32 {
    (0.5f64 * x as f64 * (1.0 + libm::erf(x as f64 / std::f64::consts::SQRT_2))) as f32
}

fn ln(x: &mut [f32], w: &[f32], b: &[f32]) {
    let n = x.len() as f64;
    let mean = x.iter().map(|v| *v as f64).sum::<f64>() / n;
    let var = x.iter().map(|v| { let d = *v as f64 - mean; d * d }).sum::<f64>() / n;
    let inv = 1.0 / (var + 1e-5f64).sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((*v as f64 - mean) * inv * w[i] as f64 + b[i] as f64) as f32;
    }
}

fn linear(x: &[f32], w: &[f32], b: &[f32], out: usize, inp: usize) -> Vec<f32> {
    let mut y = vec![0f32; out];
    for o in 0..out {
        let row = &w[o * inp..o * inp + inp];
        let mut acc = b[o] as f64;
        for (i, xv) in row.iter().enumerate() {
            acc += *xv as f64 * x[i] as f64;
        }
        y[o] = acc as f32;
    }
    y
}

pub struct Bart {
    emb: Vec<f32>,           // [VOCAB][D_MODEL] shared
    pos_enc: Vec<f32>,       // [66][D_MODEL]
    pos_dec: Vec<f32>,
    enc_ln_emb_w: Vec<f32>,
    enc_ln_emb_b: Vec<f32>,
    enc_ln_sa_w: Vec<f32>,
    enc_ln_sa_b: Vec<f32>,
    enc_ln_ff_w: Vec<f32>,
    enc_ln_ff_b: Vec<f32>,
    enc_fc1_w: Vec<f32>,
    enc_fc1_b: Vec<f32>,
    enc_fc2_w: Vec<f32>,
    enc_fc2_b: Vec<f32>,
    enc_q_w: Vec<f32>, enc_q_b: Vec<f32>, enc_k_w: Vec<f32>, enc_k_b: Vec<f32>,
    enc_v_w: Vec<f32>, enc_v_b: Vec<f32>, enc_o_w: Vec<f32>, enc_o_b: Vec<f32>,
    dec_ln_emb_w: Vec<f32>,
    dec_ln_emb_b: Vec<f32>,
    dec_ln_sa_w: Vec<f32>,
    dec_ln_sa_b: Vec<f32>,
    dec_ln_ca_w: Vec<f32>,
    dec_ln_ca_b: Vec<f32>,
    dec_ln_ff_w: Vec<f32>,
    dec_ln_ff_b: Vec<f32>,
    dec_fc1_w: Vec<f32>,
    dec_fc1_b: Vec<f32>,
    dec_fc2_w: Vec<f32>,
    dec_fc2_b: Vec<f32>,
    sa_q_w: Vec<f32>, sa_q_b: Vec<f32>, sa_k_w: Vec<f32>, sa_k_b: Vec<f32>,
    sa_v_w: Vec<f32>, sa_v_b: Vec<f32>, sa_o_w: Vec<f32>, sa_o_b: Vec<f32>,
    ca_q_w: Vec<f32>, ca_q_b: Vec<f32>, ca_k_w: Vec<f32>, ca_k_b: Vec<f32>,
    ca_v_w: Vec<f32>, ca_v_b: Vec<f32>, ca_o_w: Vec<f32>, ca_o_b: Vec<f32>,
    logits_bias: Vec<f32>,
    pub g2i: HashMap<char, usize>,
    pub i2p: HashMap<usize, char>,
}

fn get_map<'a>(m: &'a HashMap<String, Vec<f32>>, k: &str) -> &'a Vec<f32> {
    m.get(k).unwrap_or_else(|| panic!("safetensors: missing tensor {k}"))
}

impl Bart {
    pub fn load() -> Bart {
        let raw = gunzip(BART_GZ);
        let t = safetensors_f32(&raw);
        let g2i: HashMap<char, usize> = GRAPHEMES
            .char_indices()
            .filter(|(i, _)| *i > 3)
            .map(|(i, c)| (c, i))
            .collect();
        let i2p: HashMap<usize, char> = PHONEMES
            .chars()
            .enumerate()
            .filter(|(i, _)| *i > 3)
            .map(|(i, c)| (i, c))
            .collect();
        let bias: Vec<f32> = get_map(&t, "final_logits_bias").clone();
        let b = |n: &str| get_map(&t, n).clone();
        Bart {
            emb: b("model.shared.weight"),
            pos_enc: b("model.encoder.embed_positions.weight"),
            pos_dec: b("model.decoder.embed_positions.weight"),
            enc_ln_emb_w: b("model.encoder.layernorm_embedding.weight"),
            enc_ln_emb_b: b("model.encoder.layernorm_embedding.bias"),
            enc_ln_sa_w: b("model.encoder.layers.0.self_attn_layer_norm.weight"),
            enc_ln_sa_b: b("model.encoder.layers.0.self_attn_layer_norm.bias"),
            enc_ln_ff_w: b("model.encoder.layers.0.final_layer_norm.weight"),
            enc_ln_ff_b: b("model.encoder.layers.0.final_layer_norm.bias"),
            enc_fc1_w: b("model.encoder.layers.0.fc1.weight"),
            enc_fc1_b: b("model.encoder.layers.0.fc1.bias"),
            enc_fc2_w: b("model.encoder.layers.0.fc2.weight"),
            enc_fc2_b: b("model.encoder.layers.0.fc2.bias"),
            enc_q_w: b("model.encoder.layers.0.self_attn.q_proj.weight"),
            enc_q_b: b("model.encoder.layers.0.self_attn.q_proj.bias"),
            enc_k_w: b("model.encoder.layers.0.self_attn.k_proj.weight"),
            enc_k_b: b("model.encoder.layers.0.self_attn.k_proj.bias"),
            enc_v_w: b("model.encoder.layers.0.self_attn.v_proj.weight"),
            enc_v_b: b("model.encoder.layers.0.self_attn.v_proj.bias"),
            enc_o_w: b("model.encoder.layers.0.self_attn.out_proj.weight"),
            enc_o_b: b("model.encoder.layers.0.self_attn.out_proj.bias"),
            dec_ln_emb_w: b("model.decoder.layernorm_embedding.weight"),
            dec_ln_emb_b: b("model.decoder.layernorm_embedding.bias"),
            dec_ln_sa_w: b("model.decoder.layers.0.self_attn_layer_norm.weight"),
            dec_ln_sa_b: b("model.decoder.layers.0.self_attn_layer_norm.bias"),
            dec_ln_ca_w: b("model.decoder.layers.0.encoder_attn_layer_norm.weight"),
            dec_ln_ca_b: b("model.decoder.layers.0.encoder_attn_layer_norm.bias"),
            dec_ln_ff_w: b("model.decoder.layers.0.final_layer_norm.weight"),
            dec_ln_ff_b: b("model.decoder.layers.0.final_layer_norm.bias"),
            dec_fc1_w: b("model.decoder.layers.0.fc1.weight"),
            dec_fc1_b: b("model.decoder.layers.0.fc1.bias"),
            dec_fc2_w: b("model.decoder.layers.0.fc2.weight"),
            dec_fc2_b: b("model.decoder.layers.0.fc2.bias"),
            sa_q_w: b("model.decoder.layers.0.self_attn.q_proj.weight"),
            sa_q_b: b("model.decoder.layers.0.self_attn.q_proj.bias"),
            sa_k_w: b("model.decoder.layers.0.self_attn.k_proj.weight"),
            sa_k_b: b("model.decoder.layers.0.self_attn.k_proj.bias"),
            sa_v_w: b("model.decoder.layers.0.self_attn.v_proj.weight"),
            sa_v_b: b("model.decoder.layers.0.self_attn.v_proj.bias"),
            sa_o_w: b("model.decoder.layers.0.self_attn.out_proj.weight"),
            sa_o_b: b("model.decoder.layers.0.self_attn.out_proj.bias"),
            ca_q_w: b("model.decoder.layers.0.encoder_attn.q_proj.weight"),
            ca_q_b: b("model.decoder.layers.0.encoder_attn.q_proj.bias"),
            ca_k_w: b("model.decoder.layers.0.encoder_attn.k_proj.weight"),
            ca_k_b: b("model.decoder.layers.0.encoder_attn.k_proj.bias"),
            ca_v_w: b("model.decoder.layers.0.encoder_attn.v_proj.weight"),
            ca_v_b: b("model.decoder.layers.0.encoder_attn.v_proj.bias"),
            ca_o_w: b("model.decoder.layers.0.encoder_attn.out_proj.weight"),
            ca_o_b: b("model.decoder.layers.0.encoder_attn.out_proj.bias"),
            logits_bias: bias,
            g2i,
            i2p,
        }
    }

    fn embed(&self, id: usize) -> [f32; D_MODEL] {
        let mut out = [0f32; D_MODEL];
        out.copy_from_slice(&self.emb[id * D_MODEL..(id + 1) * D_MODEL]);
        out
    }

    fn attn(
        &self,
        x: &[Vec<f32>],
        q_w: &[f32], q_b: &[f32], k_w: &[f32], k_b: &[f32], v_w: &[f32], v_b: &[f32], o_w: &[f32], o_b: &[f32],
        kv: &[Vec<f32>],
        causal: bool,
    ) -> Vec<Vec<f32>> {
        let n = x.len();
        let m = kv.len();
        let q: Vec<Vec<f32>> = x.iter().map(|h| linear(h, q_w, q_b, D_MODEL, D_MODEL)).collect();
        let k: Vec<Vec<f32>> = kv.iter().map(|h| linear(h, k_w, k_b, D_MODEL, D_MODEL)).collect();
        let v: Vec<Vec<f32>> = kv.iter().map(|h| linear(h, v_w, v_b, D_MODEL, D_MODEL)).collect();
        let scale = 1f32 / (D_MODEL as f32).sqrt();
        let mut out = vec![vec![0f32; D_MODEL]; n];
        for i in 0..n {
            let mut scores = vec![0f32; m];
            let limit = if causal { i + 1 } else { m };
            for j in 0..limit {
                let mut acc = 0f32;
                for d in 0..D_MODEL {
                    acc += q[i][d] * k[j][d];
                }
                scores[j] = acc * scale;
            }
            let mx = scores[..limit].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for s in scores[..limit].iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            for j in 0..limit {
                let p = scores[j] / sum;
                for d in 0..D_MODEL {
                    out[i][d] += p * v[j][d];
                }
            }
        }
        out.iter().map(|h| linear(h, o_w, o_b, D_MODEL, D_MODEL)).collect()
    }

    fn encode(&self, ids: &[usize]) -> Vec<Vec<f32>> {
        let n = ids.len();
        let mut x: Vec<Vec<f32>> = Vec::with_capacity(n);
        for (i, id) in ids.iter().enumerate() {
            let mut h = self.embed(*id);
            for d in 0..D_MODEL {
                h[d] += self.pos_enc[(i + 2) * D_MODEL + d];
            }
            x.push(h.to_vec());
        }
        for h in x.iter_mut() {
            ln(h, &self.enc_ln_emb_w, &self.enc_ln_emb_b);
        }
        // self attention (pre-post LN as transformers 5.x does it)
        let a = self.attn(&x, &self.enc_q_w, &self.enc_q_b, &self.enc_k_w, &self.enc_k_b,
            &self.enc_v_w, &self.enc_v_b, &self.enc_o_w, &self.enc_o_b, &x, false);
        for i in 0..n {
            for d in 0..D_MODEL {
                x[i][d] += a[i][d];
            }
            ln(&mut x[i], &self.enc_ln_sa_w, &self.enc_ln_sa_b);
        }
        for i in 0..n {
            let residual = x[i].clone();
            let h: Vec<f32> = linear(&x[i], &self.enc_fc1_w, &self.enc_fc1_b, FFN, D_MODEL)
                .into_iter()
                .map(gelu)
                .collect();
            let h = linear(&h, &self.enc_fc2_w, &self.enc_fc2_b, D_MODEL, FFN);
            for d in 0..D_MODEL {
                x[i][d] = residual[d] + h[d];
            }
            ln(&mut x[i], &self.enc_ln_ff_w, &self.enc_ln_ff_b);
        }
        x
    }

    /// word -> phonemes in the model's alphabet (the same ɾ->T / ʔ->t fold the python path applies).
    pub fn phonemes(&self, word: &str) -> String {
        self.phonemes_dbg(word, false)
    }

    pub fn phonemes_dbg(&self, word: &str, debug: bool) -> String {
        let step = MAX_LEN - 2;
        let chars: Vec<char> = word.chars().collect();
        let mut out = String::new();
        for chunk in chars.chunks(step) {
            // the encoder input is <s> + graphemes + </s>, exactly as the training/reference path builds it
            let mut ids: Vec<usize> = vec![BOS];
            ids.extend(chunk.iter().filter_map(|c| self.g2i.get(c).copied()));
            ids.push(EOS);
            if ids.len() <= 2 {
                continue;
            }
            let enc = self.encode(&ids);
            if debug {
                eprint!("enc[0..4]: ");
                for v in &enc[0][..4] {
                    eprint!("{v:.5} ");
                }
                eprintln!("(n={})", enc.len());
            }
            let toks = self.decode_dbg(&enc, MAX_LEN, debug);
            for t in toks {
                if let Some(c) = self.i2p.get(&t) {
                    out.push(*c);
                }
            }
        }
        out.replace('\u{27e}', "T").replace('\u{294}', "t")
    }

    fn decode_dbg(&self, enc: &[Vec<f32>], max_len: usize, debug: bool) -> Vec<usize> {
        let mut seq: Vec<usize> = vec![BOS];
        while seq.len() < max_len {
            let n = seq.len();
            let mut x: Vec<Vec<f32>> = Vec::with_capacity(n);
            for (i, id) in seq.iter().enumerate() {
                let mut h = self.embed(*id);
                for d in 0..D_MODEL {
                    h[d] += self.pos_dec[(i + 2) * D_MODEL + d];
                }
                x.push(h.to_vec());
            }
            for h in x.iter_mut() {
                ln(h, &self.dec_ln_emb_w, &self.dec_ln_emb_b);
            }
            let a = self.attn(&x, &self.sa_q_w, &self.sa_q_b, &self.sa_k_w, &self.sa_k_b,
                &self.sa_v_w, &self.sa_v_b, &self.sa_o_w, &self.sa_o_b, &x, true);
            for i in 0..n {
                for d in 0..D_MODEL {
                    x[i][d] += a[i][d];
                }
                ln(&mut x[i], &self.dec_ln_sa_w, &self.dec_ln_sa_b);
            }
            let a = self.attn(&x, &self.ca_q_w, &self.ca_q_b, &self.ca_k_w, &self.ca_k_b,
                &self.ca_v_w, &self.ca_v_b, &self.ca_o_w, &self.ca_o_b, enc, false);
            for i in 0..n {
                for d in 0..D_MODEL {
                    x[i][d] += a[i][d];
                }
                ln(&mut x[i], &self.dec_ln_ca_w, &self.dec_ln_ca_b);
            }
            for i in 0..n {
                let residual = x[i].clone();
                let h: Vec<f32> = linear(&x[i], &self.dec_fc1_w, &self.dec_fc1_b, FFN, D_MODEL)
                    .into_iter()
                    .map(gelu)
                    .collect();
                let h = linear(&h, &self.dec_fc2_w, &self.dec_fc2_b, D_MODEL, FFN);
                for d in 0..D_MODEL {
                    x[i][d] = residual[d] + h[d];
                }
                ln(&mut x[i], &self.dec_ln_ff_w, &self.dec_ln_ff_b);
            }
            let last = x.last().unwrap();
            let mut best = 0usize;
            let mut bestv = f64::NEG_INFINITY;
            let mut logits: Vec<f32> = Vec::with_capacity(VOCAB);
            for v in 0..VOCAB {
                let mut acc = self.logits_bias[v] as f64;
                for d in 0..D_MODEL {
                    acc += last[d] as f64 * self.emb[v * D_MODEL + d] as f64;
                }
                logits.push(acc as f32);
                if acc > bestv {
                    bestv = acc;
                    best = v;
                }
            }
            if debug {
                let mut order: Vec<usize> = (0..VOCAB).collect();
                order.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap());
                let top: Vec<String> = order[..3]
                    .iter()
                    .map(|v| {
                        let c = self.i2p.get(v).map(|c| c.to_string()).unwrap_or(format!("#{v}"));
                        format!("{c}({:.3})", logits[*v])
                    })
                    .collect();
                eprintln!("step {}: {}", seq.len() - 1, top.join(" "));
            }
            seq.push(best);
            if best == EOS {
                break;
            }
        }
        seq[1..].to_vec()
    }
}

/// Minimal safetensors reader: F32 tensors only (which is all the G2P model has).
pub fn safetensors_f32(bytes: &[u8]) -> HashMap<String, Vec<f32>> {
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let json = std::str::from_utf8(&bytes[8..8 + header_len]).expect("safetensors header");
    let v: serde_json::Value = serde_json::from_str(json).expect("safetensors json");
    let data = &bytes[8 + header_len..];
    let mut out = HashMap::new();
    if let Some(obj) = v.as_object() {
        for (name, meta) in obj {
            if name == "__metadata__" {
                continue;
            }
            let off = meta["data_offsets"].as_array().expect("offsets");
            let s = off[0].as_u64().unwrap() as usize;
            let e = off[1].as_u64().unwrap() as usize;
            let mut vec = Vec::with_capacity((e - s) / 4);
            for c in data[s..e].chunks_exact(4) {
                vec.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            out.insert(name.clone(), vec);
        }
    }
    out
}
