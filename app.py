import base64
import html
import math
from pathlib import Path
from urllib.parse import quote

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent
BACKGROUND_PATH = BASE_DIR / "heritage_background.jpg"

D_MODEL = 128
N_HEADS = 4
FF_DIM = 512
N_LAYERS = 2
DROPOUT = 0.1
MAX_LEN = 128
MAX_CHARS = 500

st.set_page_config(
    page_title="Vazhi Mozhi — The Path from Tamil to English",
    page_icon="🪔",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ----------------------------------------------------------------------------
# Model (identical to the architecture trained in the notebook)
# ----------------------------------------------------------------------------
def create_padding_mask(tokens, pad_id):
    return tokens == pad_id


def create_causal_mask(sequence_length, device):
    return torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device),
        diagonal=1,
    )


class AttenovaPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128, dropout=0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class AttenovaEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx, max_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.position = AttenovaPositionalEncoding(d_model, max_len, dropout)

    def forward(self, token_ids):
        return self.position(self.embedding(token_ids) * math.sqrt(self.d_model))


def scaled_dot_product_attention(Q, K, V, key_padding_mask=None, causal_mask=None):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))

    if key_padding_mask is not None:
        mask = key_padding_mask[:, None, None, :] if scores.ndim == 4 else key_padding_mask[:, None, :]
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

    if causal_mask is not None:
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, key_padding_mask=None, causal_mask=None):
        Q = self.split_heads(self.q_proj(query))
        K = self.split_heads(self.k_proj(key))
        V = self.split_heads(self.v_proj(value))

        output, weights = scaled_dot_product_attention(Q, K, V, key_padding_mask, causal_mask)

        output = output.transpose(1, 2).contiguous()
        batch_size, seq_len, _, _ = output.shape
        output = output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        return self.out_proj(self.dropout(output)), weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.network(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        attention_output, weights = self.attention(x, x, x, key_padding_mask=padding_mask)
        x = self.norm1(x + self.dropout(attention_output))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x, weights


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, target_padding_mask, source_padding_mask, causal_mask):
        self_output, self_weights = self.self_attention(
            x, x, x, key_padding_mask=target_padding_mask, causal_mask=causal_mask
        )
        x = self.norm1(x + self.dropout(self_output))
        cross_output, cross_weights = self.cross_attention(
            x, encoder_output, encoder_output, key_padding_mask=source_padding_mask
        )
        x = self.norm2(x + self.dropout(cross_output))
        x = self.norm3(x + self.dropout(self.feed_forward(x)))
        return x, self_weights, cross_weights


class AttenovaTransformer(nn.Module):
    def __init__(
        self,
        tamil_vocab_size,
        english_vocab_size,
        tamil_pad_id,
        english_pad_id,
        d_model=D_MODEL,
        num_heads=N_HEADS,
        d_ff=FF_DIM,
        num_encoder_layers=N_LAYERS,
        num_decoder_layers=N_LAYERS,
        dropout=DROPOUT,
        max_length=MAX_LEN,
    ):
        super().__init__()
        self.tamil_pad_id = tamil_pad_id
        self.english_pad_id = english_pad_id
        self.source_embedding = AttenovaEmbedding(tamil_vocab_size, d_model, tamil_pad_id, max_length, dropout)
        self.target_embedding = AttenovaEmbedding(english_vocab_size, d_model, english_pad_id, max_length, dropout)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)]
        )
        self.output_projection = nn.Linear(d_model, english_vocab_size)

    def encode(self, source_ids):
        source_padding_mask = create_padding_mask(source_ids, self.tamil_pad_id)
        x = self.source_embedding(source_ids)
        for layer in self.encoder_layers:
            x, _ = layer(x, source_padding_mask)
        return x, source_padding_mask

    def decode(self, target_ids, encoder_output, source_padding_mask):
        target_padding_mask = create_padding_mask(target_ids, self.english_pad_id)
        causal_mask = create_causal_mask(target_ids.size(1), target_ids.device)
        x = self.target_embedding(target_ids)
        for layer in self.decoder_layers:
            x, _, _ = layer(x, encoder_output, target_padding_mask, source_padding_mask, causal_mask)
        return self.output_projection(x)

    def forward(self, source_ids, target_ids):
        encoder_output, source_padding_mask = self.encode(source_ids)
        return self.decode(target_ids, encoder_output, source_padding_mask)


def locate_model_dir():
    for folder in (BASE_DIR, BASE_DIR / "attenova_deployment"):
        if all((folder / name).exists() for name in ("best_model.pt", "tamil_tokenizer.json", "english_tokenizer.json")):
            return folder
    missing = [
        name
        for name in ("best_model.pt", "tamil_tokenizer.json", "english_tokenizer.json")
        if not (BASE_DIR / name).exists()
    ]
    raise FileNotFoundError(
        "Missing file(s) next to app.py: " + ", ".join(missing)
        + ". Place them beside app.py or inside an 'attenova_deployment' folder."
    )


@st.cache_resource(show_spinner=False)
def load_assets():
    from tokenizers import Tokenizer

    folder = locate_model_dir()
    src_tok = Tokenizer.from_file(str(folder / "tamil_tokenizer.json"))
    tgt_tok = Tokenizer.from_file(str(folder / "english_tokenizer.json"))

    try:
        checkpoint = torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(folder / "best_model.pt", map_location="cpu", weights_only=False)

    ta_vocab, en_vocab = src_tok.get_vocab_size(), tgt_tok.get_vocab_size()
    if checkpoint.get("tamil_vocab_size", ta_vocab) != ta_vocab or checkpoint.get("english_vocab_size", en_vocab) != en_vocab:
        raise RuntimeError(
            "The tokenizer files do not belong to this checkpoint "
            f"(checkpoint vocab {checkpoint.get('tamil_vocab_size')}/{checkpoint.get('english_vocab_size')}, "
            f"tokenizer vocab {ta_vocab}/{en_vocab}). Use the tokenizers saved together with best_model.pt."
        )

    model = AttenovaTransformer(
        ta_vocab,
        en_vocab,
        src_tok.token_to_id("[PAD]"),
        tgt_tok.token_to_id("[PAD]"),
        d_model=checkpoint.get("d_model", D_MODEL),
        num_heads=checkpoint.get("num_heads", N_HEADS),
        d_ff=checkpoint.get("d_ff", FF_DIM),
        num_encoder_layers=checkpoint.get("num_encoder_layers", N_LAYERS),
        num_decoder_layers=checkpoint.get("num_decoder_layers", N_LAYERS),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "src_tok": src_tok,
        "tgt_tok": tgt_tok,
        "src_eos": src_tok.token_to_id("[EOS]"),
        "bos": tgt_tok.token_to_id("[BOS]"),
        "eos": tgt_tok.token_to_id("[EOS]"),
        "meta": {
            "d_model": checkpoint.get("d_model", D_MODEL),
            "heads": checkpoint.get("num_heads", N_HEADS),
            "d_ff": checkpoint.get("d_ff", FF_DIM),
            "enc": checkpoint.get("num_encoder_layers", N_LAYERS),
            "dec": checkpoint.get("num_decoder_layers", N_LAYERS),
            "params": sum(p.numel() for p in model.parameters()),
            "epoch": checkpoint.get("epoch"),
            "val_loss": checkpoint.get("validation_loss"),
        },
    }


def translate_tamil(text):
    assets = load_assets()
    model = assets["model"]

    ids = assets["src_tok"].encode(text).ids
    if len(ids) > MAX_LEN:
        ids = ids[:MAX_LEN]
        ids[-1] = assets["src_eos"]

    with torch.inference_mode():
        source = torch.tensor([ids], dtype=torch.long)
        encoder_output, source_mask = model.encode(source)
        generated = torch.tensor([[assets["bos"]]], dtype=torch.long)

        for _ in range(MAX_LEN - 1):
            logits = model.decode(generated, encoder_output, source_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if next_token.item() == assets["eos"]:
                break
            generated = torch.cat([generated, next_token], dim=1)

    return assets["tgt_tok"].decode(generated[0].tolist(), skip_special_tokens=True).strip()


# ----------------------------------------------------------------------------
# Content
# ----------------------------------------------------------------------------
NAV = [
    ("home", "Home · முகப்பு", ":material/home:"),
    ("about", "About · பற்றி", ":material/description:"),
    ("how", "How It Works · செயல்முறை", ":material/settings:"),
    ("examples", "Examples · உதாரணம்", ":material/article:"),
    ("tech", "Tech Stack · தொழில்நுட்பம்", ":material/layers:"),
    ("impact", "Impact · தாக்கம்", ":material/eco:"),
]

CARDS = [
    ("about", "📖", "About", "பற்றி", "Learn more about Vazhi Mozhi and its purpose.", "வழி மொழி பற்றி அறிக."),
    ("how", "⚙️", "How It Works", "எப்படி செயல்படுகிறது", "Understand the translation pipeline and model.", "இது எப்படி வேலை செய்கிறது என்பதை அறிக."),
    ("examples", "📄", "Examples", "உதாரணங்கள்", "Try sample Tamil sentences and their translations.", "உதாரண வாக்கியங்களைப் பார்க்கவும்."),
    ("tech", "🧱", "Tech Stack", "தொழில்நுட்பங்கள்", "Explore the tools and technologies used.", "பயன்படுத்தப்பட்ட தொழில்நுட்பங்களை அறிக."),
    ("impact", "🌿", "Impact", "தாக்கம்", "See the potential and real-world benefits.", "இது வழங்கும் நன்மைகளைப் பார்க்கவும்."),
]

EXAMPLES = [
    ("இன்று ஒரு அழகான நாள்.", "Today is a beautiful day."),
    ("எனக்கு தமிழ் மிகவும் பிடிக்கும்.", "I like Tamil very much."),
    ("நீங்கள் எப்படி இருக்கிறீர்கள்?", "How are you?"),
    ("நான் நாளை பள்ளிக்குச் செல்வேன்.", "I will go to school tomorrow."),
    ("புத்தகம் மேஜை மீது உள்ளது.", "The book is on the table."),
]

STEPS = [
    ("Enter Tamil", "You type a sentence in Tamil.",
     "தமிழ் உள்ளீடு", "நீங்கள் தமிழில் ஒரு வாக்கியத்தை எழுதுகிறீர்கள்."),
    ("Tokenize", "A Byte Pair Encoding tokenizer splits the sentence into subword tokens and IDs.",
     "டோக்கன்களாகப் பிரித்தல்", "வாக்கியம் துணைச்சொல் டோக்கன்களாகவும் எண்களாகவும் மாற்றப்படுகிறது."),
    ("Encode", "The Transformer encoder reads all Tamil tokens and builds a contextual meaning.",
     "என்கோடிங்", "என்கோடர் தமிழ் டோக்கன்களைப் படித்து பொருள் சூழலை உருவாக்குகிறது."),
    ("Decode", "The decoder generates English tokens one at a time using greedy decoding.",
     "டிகோடிங்", "டிகோடர் ஆங்கில டோக்கன்களை ஒவ்வொன்றாக உருவாக்குகிறது."),
    ("Detokenize", "The English tokenizer joins the tokens back into readable text.",
     "மீண்டும் இணைத்தல்", "ஆங்கில டோக்கன்கள் படிக்கக்கூடிய வாக்கியமாக இணைக்கப்படுகின்றன."),
    ("Show result", "The translation appears in the English panel.",
     "முடிவு", "மொழிபெயர்ப்பு ஆங்கிலப் பெட்டியில் தோன்றும்."),
]

TECH = [
    ("Python", "The language behind the model, tokenizers and app logic.",
     "மாதிரி, டோக்கனைசர் மற்றும் பயன்பாட்டின் தர்க்கத்திற்கான நிரலாக்க மொழி."),
    ("PyTorch", "The deep learning framework used to build and train the model.",
     "மாதிரியை உருவாக்கவும் பயிற்றுவிக்கவும் பயன்படும் ஆழ்கற்றல் கட்டமைப்பு."),
    ("Streamlit", "Turns the Python code into this interactive web interface.",
     "பைத்தான் நிரலை இந்த வலை இடைமுகமாக மாற்றுகிறது."),
    ("Transformer", "An attention-based encoder–decoder architecture for translation.",
     "மொழிபெயர்ப்புக்கான attention அடிப்படையிலான என்கோடர்–டிகோடர் கட்டமைப்பு."),
    ("NLP", "Natural Language Processing: teaching computers to work with human language.",
     "இயற்கை மொழி செயலாக்கம்: மனித மொழியுடன் செயல்பட கணினிகளுக்குக் கற்பித்தல்."),
    ("BPE tokenization", "Custom Tamil and English tokenizers that split words into subword units.",
     "சொற்களைத் துணைச்சொல் அலகுகளாகப் பிரிக்கும் தமிழ் மற்றும் ஆங்கில டோக்கனைசர்கள்."),
]

def spec_rows():
    try:
        meta = load_assets()["meta"]
    except Exception:
        return []
    rows = [
        ("Model dimension", "மாதிரி பரிமாணம்", str(meta["d_model"])),
        ("Attention heads", "கவனத் தலைகள்", str(meta["heads"])),
        ("Feed-forward size", "ஃபீட்-ஃபார்வர்டு அளவு", str(meta["d_ff"])),
        ("Encoder / decoder layers", "என்கோடர் / டிகோடர் அடுக்குகள்", f'{meta["enc"]} / {meta["dec"]}'),
        ("Parameters", "அளவுருக்கள்", f'{meta["params"] / 1e6:.1f} million'),
    ]
    if meta.get("epoch"):
        rows.append(("Best epoch", "சிறந்த சுற்று", str(meta["epoch"])))
    if meta.get("val_loss"):
        rows.append(("Validation loss", "சரிபார்ப்பு இழப்பு", f'{meta["val_loss"]:.3f}'))
    return rows

IMPACT = [
    ("Supports Tamil speakers", "Helps people read English information, forms and services with more confidence.",
     "தமிழ் பேசுபவர்களுக்கு ஆதரவு", "ஆங்கிலத் தகவல்கள், படிவங்கள், சேவைகளை நம்பிக்கையுடன் புரிந்துகொள்ள உதவுகிறது."),
    ("Language learning", "Learners can compare a Tamil sentence with its English form side by side.",
     "மொழி கற்றல்", "தமிழ் வாக்கியத்தையும் அதன் ஆங்கில வடிவத்தையும் அருகருகே ஒப்பிட்டுக் கற்கலாம்."),
    ("Access to communication", "Makes English-language communication at school, work and online easier to reach.",
     "தொடர்புக்கான அணுகல்", "பள்ளி, பணியிடம், இணையத்தில் ஆங்கிலத் தொடர்பை எளிதாக்குகிறது."),
    ("A base for research", "A starting point for improving machine translation for a low-resource language.",
     "ஆராய்ச்சிக்கு அடித்தளம்", "வளம் குறைந்த மொழிக்கான இயந்திர மொழிபெயர்ப்பை மேம்படுத்த ஒரு தொடக்கப் புள்ளி."),
]


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
def svg_url(svg):
    return 'url("data:image/svg+xml,' + quote(svg) + '")'


def petals(cx, cy, r, n, fill, stroke):
    out = []
    for i in range(n):
        out.append(
            f'<ellipse cx="{cx}" cy="{cy - r * 0.55:.2f}" rx="{r * 0.27:.2f}" ry="{r * 0.5:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="0.7" transform="rotate({i * 360 / n:.1f} {cx} {cy})"/>'
        )
    return "".join(out)


def flower(cx, cy, r, kind):
    if kind == "jasmine":
        return petals(cx, cy, r, 6, "#FFFDF6", "#D8C28A") + f'<circle cx="{cx}" cy="{cy}" r="{r * 0.18:.2f}" fill="#E3B341"/>'
    return (
        petals(cx, cy, r, 10, "#F29A2E", "#C9701A")
        + petals(cx, cy, r * 0.62, 8, "#F7B94A", "#D98A25")
        + f'<circle cx="{cx}" cy="{cy}" r="{r * 0.14:.2f}" fill="#B5523B"/>'
    )


def svg_wrap(width, height, body):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">{body}</svg>'


def garland_svg():
    parts = ['<path d="M0 6 Q35 32 70 6 Q105 32 140 6" fill="none" stroke="#8A6A3C" stroke-width="1.4"/>']
    for x in (17.5, 52.5, 87.5, 122.5):
        parts.append(f'<ellipse cx="{x}" cy="21" rx="2.6" ry="6" fill="#5E8C61" transform="rotate(-20 {x} 21)"/>')
    for x, y, r, kind in [
        (0, 6, 6, "jasmine"), (70, 6, 6, "jasmine"), (140, 6, 6, "jasmine"),
        (35, 19, 9, "marigold"), (105, 19, 9, "marigold"),
        (17.5, 15.5, 4.5, "jasmine"), (52.5, 15.5, 4.5, "jasmine"),
        (87.5, 15.5, 4.5, "jasmine"), (122.5, 15.5, 4.5, "jasmine"),
    ]:
        parts.append(flower(x, y, r, kind))
    return svg_wrap(140, 36, "".join(parts))


def cluster_svg():
    parts = [
        f'<ellipse cx="55" cy="22" rx="7" ry="20" fill="#5E8C61" transform="rotate({rot} 55 45)"/>'
        for rot in (-100, -60, -20, 20, 60, 100)
    ]
    parts += [
        flower(44, 42, 17, "marigold"),
        flower(74, 52, 14, "marigold"),
        flower(26, 62, 10, "jasmine"),
        flower(86, 28, 10, "jasmine"),
        flower(58, 72, 8, "jasmine"),
    ]
    return svg_wrap(110, 90, "".join(parts))


def kolam_svg(color, size, opacity):
    c = size / 2
    parts = []
    for i in range(16):
        a = math.radians(i * 22.5)
        parts.append(f'<circle cx="{c + math.cos(a) * size * 0.46:.2f}" cy="{c + math.sin(a) * size * 0.46:.2f}" r="{size * 0.014:.2f}" fill="{color}"/>')
    for i in range(8):
        parts.append(
            f'<ellipse cx="{c}" cy="{c - size * 0.2:.2f}" rx="{size * 0.075:.2f}" ry="{size * 0.17:.2f}" fill="none" '
            f'stroke="{color}" stroke-width="{size * 0.011:.2f}" transform="rotate({i * 45} {c} {c})"/>'
        )
    for i in range(8):
        a = math.radians(i * 45 + 22.5)
        parts.append(
            f'<circle cx="{c + math.cos(a) * size * 0.36:.2f}" cy="{c + math.sin(a) * size * 0.36:.2f}" r="{size * 0.055:.2f}" '
            f'fill="none" stroke="{color}" stroke-width="{size * 0.009:.2f}"/>'
        )
    parts.append(f'<circle cx="{c}" cy="{c}" r="{size * 0.3:.2f}" fill="none" stroke="{color}" stroke-width="{size * 0.007:.2f}"/>')
    parts.append(f'<circle cx="{c}" cy="{c}" r="{size * 0.03:.2f}" fill="{color}"/>')
    return svg_wrap(size, size, f'<g opacity="{opacity}">{"".join(parts)}</g>')


@st.cache_data(show_spinner=False)
def background_css():
    if BACKGROUND_PATH.exists():
        encoded = base64.b64encode(BACKGROUND_PATH.read_bytes()).decode()
        image = f'url("data:image/jpeg;base64,{encoded}")'
    else:
        image = "linear-gradient(135deg,#F3DFC1,#E8C9A0)"
    return (
        ".stApp{background-image:"
        f"{svg_url(garland_svg())},"
        f"{svg_url(kolam_svg('#FFFFFF', 240, 0.6))},"
        f"{svg_url(kolam_svg('#B5523B', 240, 0.22))},"
        "linear-gradient(90deg,rgba(255,246,230,.06) 0%,rgba(255,246,230,.42) 50%,rgba(255,246,230,.78) 100%),"
        f"{image};"
        "background-repeat:repeat-x,no-repeat,no-repeat,no-repeat,no-repeat;"
        "background-position:left top,left -60px bottom -60px,right -70px top -70px,center,center;"
        "background-size:auto 40px,260px 260px,260px 260px,100% 100%,cover;"
        "background-attachment:fixed;}"
    )


@st.cache_data(show_spinner=False)
def decor_css():
    garland = svg_url(garland_svg())
    cluster = svg_url(cluster_svg())
    kolam = svg_url(kolam_svg("#B5523B", 130, 0.3))
    icon = svg_url(svg_wrap(40, 40, flower(20, 20, 17, "jasmine")))
    return (
        ".st-key-translator,.st-key-panel,.st-key-introbox{"
        f"background-image:{kolam};background-repeat:no-repeat;"
        "background-position:right 18px bottom 14px;background-size:130px 130px;}"
        ".st-key-translator::before,.st-key-panel::before,.st-key-introbox::before{"
        'content:"";position:absolute;top:-19px;left:56px;right:150px;height:36px;pointer-events:none;'
        f"background:{garland} repeat-x left center/auto 36px;}}"
        ".st-key-translator::after,.st-key-panel::after,.st-key-introbox::after{"
        'content:"";position:absolute;top:-30px;right:-14px;width:120px;height:98px;pointer-events:none;'
        f"background:{cluster} no-repeat center/contain;}}"
        '[class*="st-key-card_"]{'
        f"background-image:{icon};background-repeat:no-repeat;"
        "background-position:right 12px top 12px;background-size:30px 30px;}"
    )


BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Noto+Sans+Tamil:wght@400;500;600&family=Nunito+Sans:wght@400;600;700&display=swap');
:root{--cream:#FBF3E6;--ivory:#FFF9F0;--maroon:#7A1B30;--terra:#B5523B;--brown:#4A3426;--gold:#C9A24D;--sage:#3F6B5A;--line:#E8CFA9;}
html,body,[class*="css"]{font-family:'Nunito Sans','Noto Sans Tamil',sans-serif;color:var(--brown);}
header[data-testid="stHeader"],footer,#MainMenu,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important;}
.block-container{max-width:1280px;padding:2.8rem 2rem 3rem;}
.ta{font-family:'Noto Sans Tamil','Nunito Sans',sans-serif;}

.brand{display:flex;align-items:center;gap:.9rem;}
.logo{width:62px;height:62px;border-radius:18px;background:linear-gradient(145deg,#8B2236,#B5523B);color:#FFF4E2;
 display:flex;align-items:center;justify-content:center;font-family:'Noto Sans Tamil',sans-serif;font-weight:600;font-size:1.45rem;
 box-shadow:0 6px 16px rgba(122,27,48,.28);flex-shrink:0;}
.brand-name{font-family:'Playfair Display',serif;font-weight:700;font-size:1.9rem;color:var(--maroon);letter-spacing:.04em;line-height:1.1;}
.brand-sub{font-size:.82rem;color:#5B4636;}

[class*="st-key-nav_"] button{background:transparent!important;border:none!important;box-shadow:none!important;color:var(--brown)!important;
 font-weight:600;border-radius:999px!important;padding:.55rem .6rem!important;white-space:nowrap;border-bottom:2px solid transparent!important;}
[class*="st-key-nav_"] button:hover{background:rgba(255,249,240,.75)!important;color:var(--maroon)!important;}

.hero h1{font-family:'Playfair Display',serif;font-weight:800;font-size:clamp(2.6rem,6vw,4.7rem);line-height:1.02;color:var(--maroon);margin:2.2rem 0 .9rem;}
.hero h1 em{color:var(--terra);font-style:normal;}
.hero .tag{font-family:'Playfair Display',serif;font-size:1.45rem;font-weight:600;color:var(--sage);margin:0 0 .4rem;}
.hero p{max-width:640px;font-size:1.12rem;line-height:1.55;color:#4A3A2E;margin:0 0 .5rem;}

.st-key-translator,.st-key-panel,.st-key-introbox{background:rgba(255,249,240,.9);border:1.5px solid var(--line);border-radius:30px;
 padding:1.6rem 1.9rem 1.4rem;box-shadow:0 14px 40px rgba(110,60,30,.18);backdrop-filter:blur(8px);position:relative;}

.pane-head{display:flex;align-items:center;gap:.8rem;margin-bottom:.7rem;}
.pane-icon{width:44px;height:44px;border-radius:13px;background:linear-gradient(145deg,#8B2236,#B5523B);color:#FFF4E2;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:1.05rem;flex-shrink:0;}
.pane-title{font-family:'Playfair Display',serif;font-weight:700;font-size:1.3rem;color:var(--maroon);line-height:1.15;}
.pane-sub{font-size:.88rem;color:#6B5545;}

.stTextArea textarea{background:rgba(255,252,246,.96)!important;border:1px solid #E6CDAA!important;border-radius:16px!important;
 font-family:'Noto Sans Tamil','Nunito Sans',sans-serif!important;font-size:1.08rem!important;color:var(--brown)!important;padding:1rem 1.1rem!important;}
.stTextArea [data-baseweb="textarea"],.stTextArea [data-baseweb="base-input"]{background:transparent!important;border:none!important;border-radius:16px!important;box-shadow:none!important;}
.stTextArea textarea:focus{border-color:var(--terra)!important;box-shadow:0 0 0 2px rgba(181,82,59,.2)!important;}
.stTextArea textarea::placeholder{color:#9A8573;}
.outbox{background:rgba(255,252,246,.96);border:1px solid #E6CDAA;border-radius:16px;min-height:130px;padding:1rem 1.1rem;font-size:1.12rem;line-height:1.55;color:var(--brown);}
.outbox.empty{color:#9A8573;}

.st-key-translate button{background:linear-gradient(100deg,#7A1B30,#B5523B)!important;color:#FFF4E2!important;border:none!important;border-radius:14px!important;
 font-weight:700;padding:.65rem 1.4rem!important;box-shadow:0 8px 18px rgba(122,27,48,.3)!important;width:100%;}
.st-key-clear button{background:rgba(255,249,240,.9)!important;color:var(--maroon)!important;border:1.5px solid var(--terra)!important;border-radius:14px!important;
 font-weight:700;padding:.65rem 1.4rem!important;width:100%;}
.st-key-translate button:hover{filter:brightness(1.08);}
.st-key-clear button:hover{background:#FFF1DF!important;}

[class*="st-key-card_"]{background:rgba(255,247,236,.92);border:1.5px solid var(--line);border-radius:22px;padding:1rem 1.1rem .9rem;
 box-shadow:0 8px 22px rgba(110,60,30,.12);min-height:210px;}
.card-top{display:flex;align-items:center;gap:.7rem;margin-bottom:.5rem;}
.card-ico{width:42px;height:42px;border-radius:12px;background:#F6DCC4;display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex-shrink:0;}
.card-title{font-family:'Playfair Display',serif;font-weight:700;font-size:1.08rem;color:var(--maroon);}
.card-en{font-size:.84rem;color:#5B4636;line-height:1.4;margin:0 0 .3rem;}
.card-ta{font-size:.82rem;color:#6B5545;line-height:1.5;margin:0 0 .5rem;}
[class*="st-key-go_"] button{width:38px;height:38px;min-height:38px;border-radius:50%!important;background:#F6D3BD!important;color:var(--maroon)!important;border:none!important;padding:0!important;font-size:1.1rem;}

.sec-title{font-family:'Playfair Display',serif;font-size:2.2rem;font-weight:800;color:var(--maroon);margin:0;}
.sec-ta{font-size:1.15rem;color:var(--sage);margin:.15rem 0 1.1rem;font-weight:600;}
.lead{font-size:1.05rem;line-height:1.65;margin:0 0 .5rem;max-width:860px;}
.lead-ta{font-size:1.02rem;line-height:1.85;color:#5B4636;margin:0 0 1rem;max-width:860px;}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1rem;margin-top:.6rem;}
.item{background:rgba(255,252,246,.95);border:1px solid var(--line);border-radius:18px;padding:1rem 1.1rem;}
.item h4{font-family:'Playfair Display',serif;color:var(--maroon);font-size:1.12rem;margin:0 0 .2rem;}
.item .h-ta{color:var(--sage);font-weight:600;font-size:.95rem;margin:0 0 .5rem;}
.item p{font-size:.93rem;line-height:1.5;margin:0 0 .35rem;}
.item p.t{color:#6B5545;line-height:1.7;}
.step-no{display:inline-flex;width:28px;height:28px;border-radius:50%;background:var(--terra);color:#FFF4E2;font-weight:700;font-size:.85rem;align-items:center;justify-content:center;margin-right:.5rem;}
.spec{display:flex;justify-content:space-between;gap:1rem;padding:.55rem 0;border-bottom:1px dashed var(--line);font-size:.95rem;}
.spec b{color:var(--maroon);}
.pair{background:rgba(255,252,246,.95);border:1px solid var(--line);border-radius:16px;padding:.8rem 1rem;margin-bottom:.35rem;}
.pair .a{font-size:1.1rem;font-weight:600;}
.pair .b{color:var(--sage);font-weight:600;}
.note{font-size:.88rem;color:#6B5545;margin-top:.8rem;}
.intro-title{font-family:'Playfair Display',serif;font-weight:800;font-size:clamp(3rem,8vw,5.6rem);color:var(--maroon);line-height:1;margin:.2rem 0 0;}
.intro-ta{font-size:clamp(1.6rem,4vw,2.4rem);color:var(--terra);font-weight:600;margin:.3rem 0 1rem;}
.chips{display:flex;flex-wrap:wrap;gap:.6rem;margin:1rem 0 1.3rem;}
.chip{background:#F6E3C8;border:1px solid var(--line);border-radius:999px;padding:.35rem .9rem;font-size:.9rem;color:var(--maroon);font-weight:600;}
.st-key-enter button{background:linear-gradient(100deg,#7A1B30,#B5523B)!important;color:#FFF4E2!important;border:none!important;border-radius:16px!important;
 font-weight:700;font-size:1.1rem;padding:.8rem 1.6rem!important;box-shadow:0 10px 22px rgba(122,27,48,.32)!important;}
.st-key-back button{background:rgba(255,249,240,.9)!important;color:var(--maroon)!important;border:1.5px solid var(--terra)!important;border-radius:14px!important;font-weight:700;}
[class*="st-key-try_"] button{background:#F6E3C8!important;color:var(--maroon)!important;border:1px solid var(--line)!important;border-radius:12px!important;font-weight:600;}
html{color-scheme:light;}
[class*="st-key-nav_"] button p{font-size:.82rem!important;white-space:nowrap;}
[data-testid="stMarkdownContainer"]{color:#4A3426!important;}
.sec-title,.stMarkdown h2.sec-title{color:#7A1B30!important;}
.sec-ta,.item .h-ta,.pair .b,.hero .tag,.tag-ta,.card-title-ta{color:#3F6B5A!important;}
.item h4,.card-title,.pane-title,.brand-name,.hero h1,.h1-ta,.intro-title{color:#7A1B30!important;}
.hero h1 em{color:#B5523B!important;}
.lead,.item p,.card-en,.spec,.pair .a,.note,.hero p{color:#4A3426!important;}
.lead-ta,.item p.t,.card-ta,.pane-sub,.brand-sub,.hero p.p-ta,.spec .ta{color:#5B4636!important;}
.h1-ta{font-size:1.35rem;font-weight:600;margin:0 0 1rem;}
.tag-ta{font-size:1.1rem;font-weight:600;margin:0 0 .6rem;}
.hero p.p-ta{font-size:1rem;margin:0 0 1.4rem;}
.card-title-ta{font-size:.82rem;font-weight:600;}
.intro-ta{color:#B5523B!important;}
.ver{font-size:.74rem;color:#7A6552!important;margin:1.6rem 0 0;opacity:.85;}
@media(max-width:900px){.block-container{padding:1rem 1rem 2rem;}.brand-name{font-size:1.5rem;}.st-key-translator,.st-key-panel,.st-key-introbox{padding:1.2rem 1rem;}}
"""


def inject_css(page):
    active = ""
    if page in {k for k, _, _ in NAV}:
        active = (
            f".st-key-nav_{page} button{{background:rgba(255,249,240,.9)!important;"
            "color:var(--maroon)!important;border-bottom:2px solid var(--maroon)!important;}"
        )
    st.markdown(f"<style>{BASE_CSS}{background_css()}{decor_css()}{active}</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# State and callbacks
# ----------------------------------------------------------------------------
def init_state():
    defaults = {
        "page": "intro",
        "tamil_input": "",
        "output": "",
        "error": "",
        "notice": "",
        "pending": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def go(page):
    st.session_state.page = page


def request_translate():
    st.session_state.pending = True
    st.session_state.error = ""
    st.session_state.notice = ""


def clear_all():
    st.session_state.tamil_input = ""
    st.session_state.output = ""
    st.session_state.error = ""
    st.session_state.notice = ""
    st.session_state.pending = False


def use_example(text):
    st.session_state.tamil_input = text
    st.session_state.output = ""
    st.session_state.error = ""
    st.session_state.notice = ""
    st.session_state.page = "home"
    st.session_state.pending = True


# ----------------------------------------------------------------------------
# Components
# ----------------------------------------------------------------------------
def render_header():
    cols = st.columns([2.5, 1.15, 1.1, 1.75, 1.5, 1.95, 1.25], vertical_alignment="center")
    with cols[0]:
        st.markdown(
            '<div class="brand"><div class="logo">வழி</div><div>'
            '<div class="brand-name">VAZHI MOZHI</div>'
            '<div class="brand-sub">The path from Tamil to English</div>'
            '<div class="brand-sub ta">தமிழிலிருந்து ஆங்கிலத்திற்கு</div></div></div>',
            unsafe_allow_html=True,
        )
    for col, (key, label, icon) in zip(cols[1:], NAV):
        with col:
            st.button(label, icon=icon, key=f"nav_{key}", on_click=go, args=(key,), use_container_width=True)


def render_hero():
    st.markdown(
        '<div class="hero"><h1>Every language<br>has a <em>story.</em></h1>'
        '<div class="h1-ta ta">ஒவ்வொரு மொழிக்கும் ஒரு கதை உண்டு.</div>'
        '<div class="tag">The path from Tamil to English.</div>'
        '<div class="tag-ta ta">தமிழிலிருந்து ஆங்கிலத்திற்கு ஒரு வழி.</div>'
        '<p>Explore Tamil-to-English translation through an experimental Transformer model built with deep learning.</p>'
        '<p class="p-ta ta">ஆழ்கற்றலால் உருவாக்கப்பட்ட சோதனை Transformer மாதிரி மூலம் தமிழிலிருந்து ஆங்கிலத்திற்கு மொழிபெயர்ப்பை ஆராயுங்கள்.</p></div>',
        unsafe_allow_html=True,
    )


def render_translator():
    with st.container(key="translator"):
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown(
                '<div class="pane-head"><div class="pane-icon ta">வழி</div><div>'
                '<div class="pane-title">Tamil Input</div>'
                '<div class="pane-sub ta">தமிழ் வாக்கியத்தை உள்ளிடுங்கள்</div></div></div>',
                unsafe_allow_html=True,
            )
            st.text_area(
                "Tamil input",
                key="tamil_input",
                height=130,
                max_chars=MAX_CHARS,
                placeholder="எ.கா., இன்று ஒரு அழகான நாள்.",
                label_visibility="collapsed",
            )
        with right:
            st.markdown(
                '<div class="pane-head"><div class="pane-icon">A</div><div>'
                '<div class="pane-title">English Translation</div>'
                '<div class="pane-sub">Translated English sentence will appear here</div>'
                '<div class="pane-sub ta">மொழிபெயர்க்கப்பட்ட ஆங்கில வாக்கியம் இங்கே தோன்றும்</div></div></div>',
                unsafe_allow_html=True,
            )
            if st.session_state.pending:
                st.session_state.pending = False
                text = st.session_state.tamil_input.strip()
                if not text:
                    st.session_state.output = ""
                    st.session_state.notice = "Enter a Tamil sentence first, then press Translate. · முதலில் தமிழ் வாக்கியத்தை உள்ளிட்டு Translate அழுத்துங்கள்."
                else:
                    with st.spinner("Translating… · மொழிபெயர்க்கிறது…"):
                        try:
                            st.session_state.output = translate_tamil(text)
                            if not st.session_state.output:
                                st.session_state.notice = "The model returned an empty result. Try another sentence. · மாதிரி வெற்று முடிவைத் தந்தது. வேறொரு வாக்கியத்தை முயற்சிக்கவும்."
                        except Exception as exc:
                            st.session_state.output = ""
                            st.session_state.error = str(exc)
            if st.session_state.output:
                st.markdown(
                    f'<div class="outbox">{html.escape(st.session_state.output)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="outbox empty">Your translation will appear here…<br><span class="ta">உங்கள் மொழிபெயர்ப்பு இங்கே தோன்றும்…</span></div>',
                    unsafe_allow_html=True,
                )

        _, b1, b2, _ = st.columns([2.2, 1.9, 1.9, 2.2])
        with b1:
            st.button("Translate · மொழிபெயர்", icon=":material/translate:", key="translate", on_click=request_translate, use_container_width=True)
        with b2:
            st.button("Clear · அழி", icon=":material/refresh:", key="clear", on_click=clear_all, use_container_width=True)

    if st.session_state.error:
        st.error(f"Translation could not run · மொழிபெயர்ப்பை இயக்க முடியவில்லை. {st.session_state.error}")
    elif st.session_state.notice:
        st.warning(st.session_state.notice)


def render_cards():
    cols = st.columns(5, gap="small")
    for col, (key, icon, title, ta_title, en, ta) in zip(cols, CARDS):
        with col:
            with st.container(key=f"card_{key}"):
                st.markdown(
                    f'<div class="card-top"><div class="card-ico">{icon}</div><div><div class="card-title">{title}</div><div class="card-title-ta ta">{ta_title}</div></div></div>'
                    f'<p class="card-en">{en}</p><p class="card-ta ta">{ta}</p>',
                    unsafe_allow_html=True,
                )
                st.button("→", key=f"go_{key}", on_click=go, args=(key,))


def section_head(title_en, title_ta):
    st.markdown(
        f'<h2 class="sec-title">{title_en}</h2><div class="sec-ta ta">{title_ta}</div>',
        unsafe_allow_html=True,
    )


def back_button():
    st.button("← Back to translator · மொழிபெயர்ப்புக்குத் திரும்பு", key="back", on_click=go, args=("home",))


def page_about():
    section_head("About Vazhi Mozhi", "வழி மொழி பற்றி")
    st.markdown(
        '<p class="lead">Vazhi Mozhi means “the path of language”. It is a Tamil-to-English translation app powered by a Transformer '
        'encoder–decoder that was built from scratch and trained on parallel Tamil–English sentences.</p>'
        '<p class="lead-ta ta">வழி மொழி என்பது தமிழிலிருந்து ஆங்கிலத்திற்கு மொழிபெயர்க்கும் ஒரு வலைப் பயன்பாடு. தமிழ்–ஆங்கில இணை வாக்கியங்களில் '
        'புதிதாகப் பயிற்றுவிக்கப்பட்ட Transformer மாதிரியால் இது இயங்குகிறது.</p>'
        '<p class="lead">It was developed to make English communication more reachable for Tamil speakers through a simple, bilingual interface. '
        'The model is experimental, so translations can be imperfect and will improve with more training and evaluation.</p>'
        '<p class="lead-ta ta">தமிழ் பேசுபவர்கள் ஆங்கில உரையாடலை எளிதாக அணுக உதவுவதே இதன் நோக்கம். இது சோதனை நிலையில் உள்ளது; '
        'மேலும் பயிற்சி மற்றும் மதிப்பீட்டால் மொழிபெயர்ப்புத் தரம் மேம்படும்.</p>',
        unsafe_allow_html=True,
    )


def page_how():
    section_head("How It Works", "இது எப்படி வேலை செய்கிறது")
    cards = "".join(
        f'<div class="item"><h4><span class="step-no">{i}</span>{en}</h4><div class="h-ta ta">{ta}</div>'
        f'<p>{den}</p><p class="t ta">{dta}</p></div>'
        for i, (en, den, ta, dta) in enumerate(STEPS, 1)
    )
    st.markdown(f'<div class="grid">{cards}</div>', unsafe_allow_html=True)


def page_examples():
    section_head("Examples", "உதாரணங்கள்")
    st.markdown(
        '<p class="lead">Pick a sentence to send it to the translator. The English lines below are reference translations; '
        'the model’s own output may differ.</p>'
        '<p class="lead-ta ta">ஒரு வாக்கியத்தைத் தேர்ந்தெடுத்தால் அது மொழிபெயர்ப்புப் பெட்டியில் சேரும். கீழே உள்ள ஆங்கிலம் குறிப்பு மொழிபெயர்ப்பு; மாதிரியின் முடிவு வேறுபடலாம்.</p>',
        unsafe_allow_html=True,
    )
    for i, (ta, en) in enumerate(EXAMPLES):
        c1, c2 = st.columns([5, 1.2], vertical_alignment="center")
        with c1:
            st.markdown(
                f'<div class="pair"><div class="a ta">{ta}</div><div class="b">{en}</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.button("Translate this · இதை மொழிபெயர்", key=f"try_{i}", on_click=use_example, args=(ta,), use_container_width=True)


def page_tech():
    section_head("Tech Stack", "தொழில்நுட்பங்கள்")
    cards = "".join(
        f'<div class="item"><h4>{name}</h4><p>{en}</p><p class="t ta">{ta}</p></div>'
        for name, en, ta in TECH
    )
    specs = "".join(
        f'<div class="spec"><span>{en} · <span class="ta">{ta}</span></span><b>{v}</b></div>'
        for en, ta, v in spec_rows()
    ) or (
        '<p class="note">Model details appear once the model files are found.<br>'
        '<span class="ta">மாதிரி கோப்புகள் கிடைத்ததும் விவரங்கள் தோன்றும்.</span></p>'
    )
    st.markdown(
        f'<div class="grid">{cards}</div>'
        '<h4 class="sec-ta" style="margin-top:1.4rem;">Model details · <span class="ta">மாதிரி விவரங்கள்</span></h4>'
        f'<div style="max-width:560px;">{specs}</div>'
        '<p class="note">Dataset: gopi30/english-tamil (Hugging Face). Decoding: greedy. Translation quality is experimental.<br>'
        '<span class="ta">தரவுத்தொகுப்பு: gopi30/english-tamil. மொழிபெயர்ப்புத் தரம் சோதனை நிலையில் உள்ளது.</span></p>',
        unsafe_allow_html=True,
    )


def page_impact():
    section_head("Impact", "தாக்கம்")
    cards = "".join(
        f'<div class="item"><h4>{en}</h4><div class="h-ta ta">{ta}</div><p>{den}</p><p class="t ta">{dta}</p></div>'
        for en, den, ta, dta in IMPACT
    )
    st.markdown(f'<div class="grid">{cards}</div>', unsafe_allow_html=True)


SECTION_PAGES = {
    "about": page_about,
    "how": page_how,
    "examples": page_examples,
    "tech": page_tech,
    "impact": page_impact,
}


def render_intro():
    st.markdown("<div style='height:3vh'></div>", unsafe_allow_html=True)
    left, _ = st.columns([1.35, 1])
    with left:
        with st.container(key="introbox"):
            st.markdown(
                '<div class="brand"><div class="logo">வழி</div><div class="brand-sub" style="font-size:.95rem;">'
                'Tamil · English · Transformer</div></div>'
                '<div class="intro-title">Vazhi Mozhi</div>'
                '<div class="intro-ta ta">வழி மொழி</div>'
                '<div class="tag" style="font-family:\'Playfair Display\',serif;font-size:1.5rem;font-weight:600;color:#3F6B5A;">'
                'The path from Tamil to English.</div>'
                '<div class="tag-ta ta">தமிழிலிருந்து ஆங்கிலத்திற்கு ஒரு வழி.</div>'
                '<p class="lead" style="margin-top:.8rem;">Type a sentence in Tamil and get its English translation from a Transformer model '
                'trained from scratch — an experimental step toward making English more reachable for Tamil speakers.</p>'
                '<p class="lead-ta ta">தமிழ் வாக்கியத்தை உள்ளிடுங்கள்; புதிதாகப் பயிற்றுவிக்கப்பட்ட Transformer மாதிரி அதன் ஆங்கில மொழிபெயர்ப்பை வழங்கும்.</p>'
                '<div class="chips"><span class="chip">Transformer</span>'
                '<span class="chip">BPE tokenizers · <span class="ta">டோக்கனைசர்கள்</span></span>'
                '<span class="chip">Bilingual · <span class="ta">இருமொழி</span></span></div>',
                unsafe_allow_html=True,
            )
            st.button("Enter Vazhi Mozhi · உள்ளே செல்லுங்கள்", icon=":material/arrow_forward:", key="enter", on_click=go, args=("home",))


def render_footer():
    st.markdown(
        '<div class="ver">Vazhi Mozhi · வழி மொழி · experimental Tamil-to-English translation · சோதனை மொழிபெயர்ப்பு</div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
def main():
    init_state()
    page = st.session_state.page
    inject_css(page)

    if page == "intro":
        render_intro()
        render_footer()
        return

    render_header()

    if page == "home":
        render_hero()
        render_translator()
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        render_cards()
    else:
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        with st.container(key="panel"):
            SECTION_PAGES[page]()
            back_button()

    render_footer()


main()
