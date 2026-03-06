use memmap2::Mmap;
use mummymaze::game::State;
use rustc_hash::FxHashMap;
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

pub struct AgentProbs {
    path: PathBuf,
    last_mtime: Option<SystemTime>,
    /// level key → (data_offset, n_states)
    index: FxHashMap<String, (usize, usize)>,
    mmap: Option<Mmap>,
    /// Cached lookup for the currently selected level.
    loaded_key: String,
    loaded_probs: FxHashMap<State, [f32; 5]>,
}

const ENTRY_SIZE: usize = 68; // 48B state (12 × i32) + 20B probs (5 × f32)

impl AgentProbs {
    pub fn new(path: PathBuf) -> Self {
        AgentProbs {
            path,
            last_mtime: None,
            index: FxHashMap::default(),
            mmap: None,
            loaded_key: String::new(),
            loaded_probs: FxHashMap::default(),
        }
    }

    /// Check mtime and re-mmap if changed. Returns true on update.
    pub fn poll(&mut self) -> bool {
        let mtime = match fs::metadata(&self.path).and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(_) => return false,
        };

        if self.last_mtime == Some(mtime) {
            return false;
        }

        let file = match fs::File::open(&self.path) {
            Ok(f) => f,
            Err(_) => return false,
        };

        // SAFETY: file is read-only, we accept stale reads from concurrent writes
        let mmap = match unsafe { Mmap::map(&file) } {
            Ok(m) => m,
            Err(_) => return false,
        };

        if mmap.len() < 16 {
            return false;
        }

        // Validate header
        if &mmap[0..4] != b"MMPR" {
            eprintln!("agent_probs.bin: bad magic");
            return false;
        }

        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != 1 {
            eprintln!("agent_probs.bin: unsupported version {version}");
            return false;
        }

        let n_levels = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;
        let index_offset = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;

        if index_offset > mmap.len() {
            eprintln!("agent_probs.bin: index_offset out of bounds");
            return false;
        }

        // Parse index section
        let mut index = FxHashMap::default();
        index.reserve(n_levels);
        let mut pos = index_offset;

        for _ in 0..n_levels {
            if pos + 2 > mmap.len() {
                break;
            }
            let key_len = u16::from_le_bytes(mmap[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;

            if pos + key_len + 8 > mmap.len() {
                break;
            }
            let key = match std::str::from_utf8(&mmap[pos..pos + key_len]) {
                Ok(s) => s.to_string(),
                Err(_) => break,
            };
            pos += key_len;

            let data_offset =
                u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()) as usize;
            let n_states =
                u32::from_le_bytes(mmap[pos + 4..pos + 8].try_into().unwrap()) as usize;
            pos += 8;

            index.insert(key, (data_offset, n_states));
        }

        self.index = index;
        self.mmap = Some(mmap);
        self.last_mtime = Some(mtime);
        // Invalidate cached level — will be rebuilt on next get_state_probs call
        self.loaded_key.clear();
        self.loaded_probs.clear();
        true
    }

    /// Look up action probabilities for a specific state in a level.
    /// Builds a per-level hashmap on first access (or when level/file changes).
    pub fn get_state_probs(&mut self, key: &str, state: &State) -> Option<[f32; 5]> {
        if self.mmap.is_none() {
            return None;
        }
        if !self.index.contains_key(key) {
            return None;
        }

        // Rebuild cache if level changed
        if self.loaded_key != key {
            self.load_level(key);
        }

        self.loaded_probs.get(state).copied()
    }

    /// Parse all states for one level from the mmap into a hashmap.
    fn load_level(&mut self, key: &str) {
        self.loaded_key.clear();
        self.loaded_probs.clear();

        let Some(mmap) = self.mmap.as_ref() else {
            return;
        };
        let Some(&(data_offset, n_states)) = self.index.get(key) else {
            return;
        };

        let slice_end = data_offset + n_states * ENTRY_SIZE;
        if slice_end > mmap.len() {
            return;
        }

        let data = &mmap[data_offset..slice_end];
        self.loaded_probs.reserve(n_states);

        for i in 0..n_states {
            let base = i * ENTRY_SIZE;

            // Parse 12 i32 fields into State
            let mut t = [0i32; 12];
            for (j, val) in t.iter_mut().enumerate() {
                let off = base + j * 4;
                *val = i32::from_le_bytes(data[off..off + 4].try_into().unwrap());
            }
            let state = State::from_i32_array(&t);

            // Parse 5 f32 probabilities
            let prob_base = base + 48;
            let mut probs = [0.0f32; 5];
            for (k, p) in probs.iter_mut().enumerate() {
                let off = prob_base + k * 4;
                *p = f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
            }

            self.loaded_probs.insert(state, probs);
        }

        self.loaded_key = key.to_string();
    }
}
