//! Minimal reader for the voice packs: a .npz is a ZIP of .npy arrays (numpy writes them STORED).
//! Only what we need: name -> f32 array with its shape.

pub struct Npy {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

struct Entry<'a> {
    name: String,
    data: &'a [u8],
}

fn zip_entries(blob: &[u8]) -> Vec<Entry<'_>> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 30 <= blob.len() {
        if blob[i..i + 4] != *b"PK\x03\x04" {
            i += 1;
            continue;
        }
        let method = u16::from_le_bytes([blob[i + 8], blob[i + 9]]) as usize;
        let csize = u32::from_le_bytes([blob[i + 18], blob[i + 19], blob[i + 20], blob[i + 21]]) as usize;
        let nlen = u16::from_le_bytes([blob[i + 26], blob[i + 27]]) as usize;
        let elen = u16::from_le_bytes([blob[i + 28], blob[i + 29]]) as usize;
        let start = i + 30 + nlen + elen;
        if start + csize > blob.len() {
            break;
        }
        let name = String::from_utf8_lossy(&blob[i + 30..i + 30 + nlen]).to_string();
        let data = &blob[start..start + csize];
        // Only STORED (method 0) is supported; numpy's savez uses it.
        out.push(Entry {
            name,
            data: if method == 0 { data } else { &[] },
        });
        i = start + csize;
    }
    out
}

/// Parse one .npy payload (little-endian f4 only).
pub fn npy_f32(payload: &[u8]) -> Option<Npy> {
    if payload.len() < 10 || payload[..6] != *b"\x93NUMPY" {
        return None;
    }
    let major = payload[6];
    let hdr_len = if major == 1 {
        u16::from_le_bytes([payload[8], payload[9]]) as usize
    } else {
        u32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]) as usize
    };
    let hdr_start = if major == 1 { 10 } else { 12 };
    let hdr = std::str::from_utf8(&payload[hdr_start..hdr_start + hdr_len]).ok()?;
    if !hdr.contains("<f4") {
        return None;
    }
    // shape: (a, b)
    let shp = hdr.find("'/shape'").or(hdr.find("'shape'"))?;
    let rest = &hdr[shp..];
    let open = rest.find('(')?;
    let close = rest.find(')')?;
    let shape: Vec<usize> = rest[open + 1..close]
        .split(',')
        .map(|s| s.trim().parse::<usize>().unwrap_or(0))
        .filter(|v| *v > 0)
        .collect();
    let n: usize = shape.iter().product();
    let data_start = hdr_start + hdr_len;
    if payload.len() < data_start + n * 4 {
        return None;
    }
    let data: Vec<f32> = payload[data_start..data_start + n * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some(Npy { shape, data })
}

/// All voices of one language from an npz blob: name -> (rows, cols, data).
pub fn voices_from_npz(blob: &[u8]) -> Vec<(String, Vec<f32>, usize)> {
    let mut out = Vec::new();
    for e in zip_entries(blob) {
        let name = e.name.trim_end_matches(".npy").to_string();
        if let Some(npy) = npy_f32(e.data) {
            let cols = npy.shape.last().copied().unwrap_or(0);
            out.push((name, npy.data, cols));
        }
    }
    out
}
