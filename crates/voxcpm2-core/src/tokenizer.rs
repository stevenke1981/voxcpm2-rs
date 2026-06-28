use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::Path};
use tokenizers::Tokenizer;

// ---------------------------------------------------------------------------
// Special tokens for VoxCPM2
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    pub unk_token: u32,
    pub bos_token: u32,
    pub eos_token: u32,
    pub im_start: u32,
    pub im_end: u32,
    pub audio_start: u32,
    pub audio_end: u32,
    pub audio_prompt_start: Option<u32>,
    pub audio_prompt_end: Option<u32>,
    pub ref_audio_start: u32,
    pub ref_audio_end: u32,
}

impl SpecialTokens {
    /// Load from tokenizer_config.json and the actual tokenizer vocabulary.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let cfg_path = model_dir.as_ref().join("tokenizer_config.json");
        let cfg_text = std::fs::read_to_string(&cfg_path)
            .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", cfg_path.display()))?;
        let cfg: serde_json::Value = serde_json::from_str(&cfg_text)?;

        // Read tokenizer to resolve actual IDs
        let tok_path = model_dir.as_ref().join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("cannot load {}: {e}", tok_path.display()))?;

        let resolve = |name: &str| -> anyhow::Result<u32> {
            let token_str = cfg
                .get(name)
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("{name} not in tokenizer_config.json"))?;
            tokenizer
                .token_to_id(token_str)
                .ok_or_else(|| anyhow::anyhow!("{name}={token_str} not found in vocabulary"))
        };

        let resolve_custom = |token_str: &str| -> anyhow::Result<u32> {
            tokenizer
                .token_to_id(token_str)
                .ok_or_else(|| anyhow::anyhow!("custom token {token_str} not found in vocabulary"))
        };

        Ok(SpecialTokens {
            unk_token: resolve("unk_token")?,
            bos_token: resolve("bos_token")?,
            eos_token: resolve_custom("<|im_end|>")?, // VoxCPM2 uses <|im_end|> as EOS
            im_start: resolve_custom("<|im_start|>")?,
            im_end: resolve_custom("<|im_end|>")?,
            audio_start: resolve_custom("<|audio_start|>")?,
            audio_end: resolve_custom("<|audio_end|>")?,
            audio_prompt_start: tokenizer.token_to_id("<|audio_prompt_start|>"),
            audio_prompt_end: tokenizer.token_to_id("<|audio_prompt_end|>"),
            ref_audio_start: resolve_custom("<|ref_audio_start|>")?,
            ref_audio_end: resolve_custom("<|ref_audio_end|>")?,
        })
    }
}

// ---------------------------------------------------------------------------
// CJK multi-character split map
// ---------------------------------------------------------------------------

/// Build a map from token-id → list-of-character-ids for multi-character CJK
/// tokens.  This mirrors `tokenization_voxcpm2.py` `_build_split_map()`.
pub fn build_cjk_split_map(tokenizer: &Tokenizer) -> HashMap<u32, Vec<u32>> {
    let vocab = tokenizer.get_vocab(true);
    let mut map: HashMap<u32, Vec<u32>> = HashMap::new();

    for (token, &tid) in vocab.iter() {
        let clean = token.replace('\u{2581}', ""); // remove leading space marker
        if clean.len() >= 2 && clean.chars().all(is_cjk) {
            let char_ids: Vec<u32> = clean
                .chars()
                .filter_map(|c| {
                    let s = c.to_string();
                    tokenizer.token_to_id(&s)
                })
                .collect();
            if char_ids.len() == clean.chars().count()
                && char_ids
                    .iter()
                    .all(|&id| id != tokenizer.token_to_id("<unk>").unwrap_or(0))
            {
                map.insert(tid, char_ids);
            }
        }
    }
    map
}

fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}'
        | '\u{3400}'..='\u{4DBF}'
        | '\u{F900}'..='\u{FAFF}'
        | '\u{20000}'..='\u{2A6DF}'
    )
}

// ---------------------------------------------------------------------------
// Chat template (simple renderer for VoxCPM2's specific template)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Renders VoxCPM2's chat template without a full Jinja engine.
///
/// Template:
/// ```text
/// {% for message in messages %}
/// {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
/// {% endfor %}
/// {% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
/// ```
pub fn render_chat_template(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str("<|im_start|>");
        out.push_str(&msg.role);
        out.push('\n');
        out.push_str(&msg.content);
        out.push_str("<|im_end|>\n");
    }
    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    out
}

// ---------------------------------------------------------------------------
// VoxTokenizer — main public API
// ---------------------------------------------------------------------------

pub struct VoxTokenizer {
    inner: Tokenizer,
    special: SpecialTokens,
    cjk_split_map: HashMap<u32, Vec<u32>>,
    add_bos: bool,
}

impl VoxTokenizer {
    /// Load from model directory.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let tok_path = model_dir.as_ref().join("tokenizer.json");
        let inner = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("failed to load {}: {e}", tok_path.display()))?;
        let special = SpecialTokens::from_model_dir(model_dir)?;
        let cjk_split_map = build_cjk_split_map(&inner);
        let add_bos = true; // from tokenizer_config.json add_bos_token = true

        tracing::info!(
            "VoxTokenizer loaded: vocab_size={}, cjk_splits={}, bos={}, eos={}",
            inner.get_vocab(true).len(),
            cjk_split_map.len(),
            special.bos_token,
            special.eos_token,
        );

        Ok(Self {
            inner,
            special,
            cjk_split_map,
            add_bos,
        })
    }

    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special
    }

    /// Encode text with CJK multi-char expansion, no special tokens added.
    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
        let ids = enc.get_ids().to_vec();
        Ok(self.expand_cjk(&ids))
    }

    /// Encode text with BOS/EOS according to config.
    pub fn encode_with_special(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let mut ids = Vec::new();
        if self.add_bos {
            ids.push(self.special.bos_token);
        }
        ids.extend(self.encode(text)?);
        // VoxCPM2 does NOT add EOS by default (add_eos_token = false)
        Ok(ids)
    }

    /// Encode chat messages using the chat template, then tokenize.
    pub fn encode_chat(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> anyhow::Result<Vec<u32>> {
        let text = render_chat_template(messages, add_generation_prompt);
        self.encode_with_special(&text)
    }

    /// Encode official VoxCPM2 zero-shot/design text tokens.
    ///
    /// Python `VoxCPM2Model._generate()` uses:
    /// `text_tokenizer(target_text)` followed by `<|audio_start|>`, with no BOS,
    /// no EOS, and no chat template.
    pub fn encode_zero_shot(&self, target_text: &str) -> anyhow::Result<Vec<u32>> {
        let mut ids = self.encode(target_text)?;
        ids.push(self.special.audio_start);
        Ok(ids)
    }

    /// Apply CJK split expansion to existing IDs.
    pub fn expand_cjk(&self, ids: &[u32]) -> Vec<u32> {
        let mut result = Vec::new();
        for &tid in ids {
            match self.cjk_split_map.get(&tid) {
                Some(expansion) => result.extend_from_slice(expansion),
                None => result.push(tid),
            }
        }
        result
    }

    /// Decode IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))
    }

    /// Expose the inner tokenizer for advanced usage.
    pub fn inner(&self) -> &Tokenizer {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn get_model_dir() -> std::path::PathBuf {
        // Honor VOXCPM2_MODEL_DIR env var, otherwise skip model-dependent tests
        std::env::var("VOXCPM2_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| "models/VoxCPM2".into())
    }

    #[test]
    #[ignore = "requires model weights"]
    fn tokenizer_basic_encode() {
        let dir = get_model_dir();
        if !dir.join("tokenizer.json").exists() {
            eprintln!("SKIP: missing {}", dir.join("tokenizer.json").display());
            return;
        }
        let tok = VoxTokenizer::from_model_dir(&dir).unwrap();
        let ids = tok.encode("hello world").unwrap();
        assert!(!ids.is_empty());
        let decoded = tok.decode(&ids).unwrap();
        assert_eq!(decoded, "hello world");
    }

    #[test]
    #[ignore = "requires model weights"]
    fn tokenizer_cjk_splitting() {
        let dir = get_model_dir();
        if !dir.join("tokenizer.json").exists() {
            eprintln!("SKIP: missing tokenizer.json");
            return;
        }
        let tok = VoxTokenizer::from_model_dir(&dir).unwrap();

        // Without CJK splitting, 你好 could be a single multi-char token.
        // With splitting, each character should be separate.
        let ids = tok.encode("你好").unwrap();
        assert!(!ids.is_empty(), "你好 should produce at least one token");

        // Decode should roundtrip
        let decoded = tok.decode(&ids).unwrap();
        assert_eq!(decoded, "你好", "roundtrip: 你好");

        // Check that BOS is not added in encode()
        assert_ne!(
            ids.first(),
            Some(&tok.special.bos_token),
            "encode() should not add BOS"
        );
    }

    #[test]
    #[ignore = "requires model weights"]
    fn tokenizer_bos() {
        let dir = get_model_dir();
        if !dir.join("tokenizer.json").exists() {
            eprintln!("SKIP: missing tokenizer.json");
            return;
        }
        let tok = VoxTokenizer::from_model_dir(&dir).unwrap();
        let ids = tok.encode_with_special("test").unwrap();
        assert_eq!(ids[0], tok.special.bos_token, "first token should be BOS");
    }

    #[test]
    fn chat_template_simple() {
        let msgs = vec![
            ChatMessage {
                role: "user".into(),
                content: "Hello".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "Hi".into(),
            },
        ];
        let rendered = render_chat_template(&msgs, false);
        assert!(rendered.contains("<|im_start|>user"));
        assert!(rendered.contains("Hello"));
        assert!(rendered.contains("<|im_end|>"));
        // With add_generation_prompt=false, there should be no trailing <|im_start|>assistant
        assert!(
            !rendered.ends_with("<|im_start|>assistant\n"),
            "no trailing assistant prompt"
        );

        let with_prompt = render_chat_template(&msgs, true);
        assert!(with_prompt.contains("<|im_start|>assistant\n"));
    }

    #[test]
    #[ignore = "requires model weights"]
    fn tokenizer_chat_roundtrip() {
        let dir = get_model_dir();
        if !dir.join("tokenizer.json").exists() {
            eprintln!("SKIP: missing tokenizer.json");
            return;
        }
        let tok = VoxTokenizer::from_model_dir(&dir).unwrap();
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: "你好".into(),
        }];
        let ids = tok.encode_chat(&msgs, true).unwrap();
        assert_eq!(ids[0], tok.special.bos_token);
        // Should contain im_start and im_end tokens
        assert!(
            ids.contains(&tok.special.im_start),
            "im_start should be in chat encoding"
        );
        assert!(
            ids.contains(&tok.special.im_end),
            "im_end should be in chat encoding"
        );
    }

    #[test]
    #[ignore = "requires model weights"]
    fn special_tokens_resolved() {
        let dir = get_model_dir();
        if !dir.join("tokenizer.json").exists() {
            eprintln!("SKIP: missing tokenizer.json");
            return;
        }
        let tok = VoxTokenizer::from_model_dir(&dir).unwrap();
        let s = tok.special_tokens();
        assert_ne!(s.unk_token, s.bos_token, "UNK and BOS should differ");
        assert_ne!(s.bos_token, s.eos_token, "BOS and EOS should differ");
        assert_ne!(s.audio_start, s.audio_end);
        // audio_prompt tokens may be missing (not in all model vocabularies)
        if let (Some(a), Some(b)) = (s.audio_prompt_start, s.audio_prompt_end) {
            assert_ne!(a, b);
        }
    }

    #[test]
    #[ignore = "requires model weights"]
    fn zero_shot_appends_audio_start_without_chat_or_bos() {
        let dir = get_model_dir();
        if !dir.join("tokenizer.json").exists() {
            eprintln!("SKIP: missing tokenizer.json");
            return;
        }
        let tok = VoxTokenizer::from_model_dir(&dir).unwrap();
        let plain = tok.encode("你好").unwrap();
        let ids = tok.encode_zero_shot("你好").unwrap();
        assert_eq!(&ids[..plain.len()], plain.as_slice());
        assert_eq!(ids.last(), Some(&tok.special.audio_start));
        assert_ne!(ids.first(), Some(&tok.special.bos_token));
        assert!(!ids.contains(&tok.special.im_start));
        assert!(!ids.contains(&tok.special.im_end));
    }

    /// ── Parity test: compare Rust tokenizer output with Python golden ──
    #[derive(serde::Deserialize)]
    #[allow(dead_code)]
    struct GoldenCase {
        text: String,
        ids_no_special: Vec<u32>,
        ids_with_special: Vec<u32>,
        decoded: String,
    }

    #[derive(serde::Deserialize)]
    #[allow(dead_code)]
    struct ChatCase {
        chat: Option<bool>,
        messages: Vec<ChatMessage>,
        ids: Vec<u32>,
    }

    #[derive(serde::Deserialize)]
    #[allow(dead_code)]
    struct GoldenFile {
        special_ids: std::collections::HashMap<String, serde_json::Value>,
        cases: Vec<serde_json::Value>,
    }

    #[test]
    #[ignore = "requires model weights; run with VOXCPM2_MODEL_DIR set"]
    fn tokenizer_parity_python() {
        let dir = get_model_dir();
        if !dir.join("tokenizer.json").exists() {
            eprintln!("SKIP: missing tokenizer.json");
            return;
        }

        // Load golden
        let golden_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("tests/golden/tokenizer_cases.json");
        if !golden_path.exists() {
            eprintln!("SKIP: golden file not found at {}", golden_path.display());
            return;
        }
        let golden_text = std::fs::read_to_string(&golden_path).expect("read golden file");
        let golden: GoldenFile = serde_json::from_str(&golden_text).expect("parse golden JSON");

        let tok = VoxTokenizer::from_model_dir(&dir).unwrap();

        let mut passed = 0usize;
        let mut failed = 0usize;

        for (i, case_val) in golden.cases.iter().enumerate() {
            // Detect chat case by presence of "chat" field
            if case_val
                .get("chat")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                // chat case
                let chat: ChatCase =
                    serde_json::from_value(case_val.clone()).expect("parse chat case");
                let rust_ids = tok.encode_chat(&chat.messages, true).unwrap();
                if rust_ids == chat.ids {
                    passed += 1;
                } else {
                    failed += 1;
                    let chat_text = render_chat_template(&chat.messages, true);
                    eprintln!(
                        "FAIL chat case {}:\n  text: {chat_text:?}\n  python: {:?}\n  rust:   {:?}",
                        i,
                        &chat.ids[..10.min(chat.ids.len())],
                        &rust_ids[..10.min(rust_ids.len())],
                    );
                    // Show first difference
                    for (j, (p, r)) in chat.ids.iter().zip(rust_ids.iter()).enumerate() {
                        if p != r {
                            eprintln!("  first diff at index {j}: python={p} rust={r}");
                            break;
                        }
                    }
                }
                continue;
            }

            // Normal case
            let case: GoldenCase =
                serde_json::from_value(case_val.clone()).expect("parse normal case");

            // Test 1: ids_no_special
            let rust_ids = tok.encode(&case.text).unwrap();
            let match_no_special = rust_ids == case.ids_no_special;

            // Test 2: ids_with_special
            let rust_ids_special = tok.encode_with_special(&case.text).unwrap();
            let match_special = rust_ids_special == case.ids_with_special;

            if match_no_special && match_special {
                passed += 1;
            } else {
                failed += 1;
                eprintln!(
                    "FAIL case {i}: text={:?}",
                    &case.text[..40.min(case.text.len())]
                );
                if !match_no_special {
                    eprintln!(
                        "  ids_no_special mismatch:\n    python: {:?}\n    rust:   {:?}",
                        &case.ids_no_special[..10.min(case.ids_no_special.len())],
                        &rust_ids[..10.min(rust_ids.len())],
                    );
                }
                if !match_special {
                    eprintln!(
                        "  ids_with_special mismatch:\n    python: {:?}\n    rust:   {:?}",
                        &case.ids_with_special[..10.min(case.ids_with_special.len())],
                        &rust_ids_special[..10.min(rust_ids_special.len())],
                    );
                }
            }
        }

        let total = passed + failed;
        eprintln!("tokenizer_parity: {passed}/{total} passed, {failed} failed");
        assert_eq!(failed, 0, "{failed} parity test(s) failed");
    }
}
