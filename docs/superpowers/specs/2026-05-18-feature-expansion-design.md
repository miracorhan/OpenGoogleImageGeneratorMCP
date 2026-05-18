# OpenGoogleImageGeneratorMCP — Feature Expansion Design

**Date:** 2026-05-18  
**Status:** Approved  
**Scope:** Yaklaşım B — Parametre zenginleştirme + yeni araçlar

---

## Bağlam ve Motivasyon

Mevcut MCP 7 araç sunuyor: `generate_image`, `edit_image`, `transform_image`, `analyze_image`, `upscale_image`, `remove_background`, `generate_video`. nanobanana-mcp-server incelemesi ve güncel Google Cloud dokümantasyonu temelinde iki kategori geliştirme tanımlandı:

1. Mevcut araçlara parametre zenginleştirmesi
2. Yeni araçlar (Veo video genişlemesi, pipeline, batch, müzik)

**Kritik bulgu:** Tüm Imagen API endpointleri 30 Haziran 2026 itibarıyla kullanımdan kalkıyor; Google'ın resmi migration hedefi `gemini-2.5-flash-image`. Model tier tasarımı bu geçişi gözetiyor.

---

## Bölüm 1 — Parametre Değişiklikleri

### 1.1 `model_tier` Soyutlaması

`model_name` korunur (geriye uyumlu). `model_tier` opsiyonel olarak tüm üretim araçlarına eklenir; ikisi birlikte verilirse `model_tier` önceliklidir. Çözüm `model_registry.py`'deki `resolve_model(tier, tool_type)` fonksiyonu üzerinden yapılır.

**`generate_image` için:**

| tier | model_name | api_backend |
|---|---|---|
| `fast` | `imagen-4.0-fast-generate-001` | imagen |
| `balanced` | `gemini-2.5-flash-image` | gemini |
| `quality` | `imagen-4.0-generate-001` | imagen |
| `ultra` | `imagen-4.0-ultra-generate-001` | imagen |

**`transform_image` / `edit_image` için:**

| tier | model_name | api_backend |
|---|---|---|
| `fast` | `gemini-2.5-flash-image` | gemini |
| `quality` | `gemini-2.5-pro-image` | gemini |

> **Not:** `gemini-2.5-pro-image` runtime'da `tool_list_available_models` ile doğrulanmalı; erişilemiyorsa `gemini-2.5-flash-image`'a düşer ve kullanıcıya bildirilir.

**`generate_video` için:**

| tier | model_name | api_backend |
|---|---|---|
| `fast` | `veo-3.1-fast-generate-001` | veo |
| `quality` | `veo-3.1-generate-001` | veo |

### 1.2 `generate_image`'e Eklenen Parametreler

Vertex AI Imagen API'sinin desteklediği ancak şu an kullanılmayan alanlar:

| Parametre | Tip | Default | Açıklama |
|---|---|---|---|
| `seed` | `Optional[int]` | `None` | Deterministik üretim; `add_watermark=False` gerektirir |
| `negative_prompt` | `Optional[str]` | `None` | İstenmeyen öğeleri dışla |
| `enhance_prompt` | `bool` | `True` | LLM ile prompt rewriting |
| `add_watermark` | `bool` | `True` | SynthID dijital watermark |
| `safety_setting` | `str` | `block_medium_and_above` | `block_low_and_above` / `block_medium_and_above` / `block_only_high` |
| `person_generation` | `str` | `allow_adult` | `allow_all` / `allow_adult` / `dont_allow` |
| `output_format` | `str` | `PNG` | `PNG` / `JPEG` |
| `compression_quality` | `int` | `85` | JPEG sıkıştırma kalitesi (0-100) |
| `storage_uri` | `Optional[str]` | `None` | Direkt Cloud Storage hedefi (örn. `gs://bucket/path/`) |
| `model_tier` | `Optional[str]` | `None` | `fast` / `balanced` / `quality` / `ultra` |

### 1.3 `output_path` — Tüm Araçlara Tam Yol Desteği

Tüm araçlara `output_path: Optional[str]` eklenir. `output_filename` opsiyonel hale gelir; **ikisinden tam olarak biri zorunludur** (ikisi birden verilirse `output_path` önceliklidir; hiçbiri verilmezse validation hatası döner).

Öncelik kuralı:
```
output_path (tam yol) > DEFAULT_OUTPUT_DIR / output_filename
```

Güvenlik validasyonu: `output_path` mutlak yol olmalı, `..` içermemeli.

```python
def _validate_output_path(path: str) -> str:
    abs_path = os.path.abspath(path)
    if ".." in path or not os.path.isabs(path):
        raise ValueError("output_path must be an absolute path without '..'")
    return abs_path
```

### 1.4 `generate_video` Parametre Genişlemesi

Mevcut araca eklenenler:

| Parametre | Tip | Geçerli değerler |
|---|---|---|
| `duration` | `int` | `4` / `6` / `8` (saniye) |
| `resolution` | `str` | `"720p"` / `"1080p"` / `"4k"` |
| `aspect_ratio` | `str` | `"16:9"` / `"9:16"` |
| `audio_enabled` | `bool` | Veo 3+ destekliyor |
| `model_tier` | `str` | `fast` / `quality` |

---

## Bölüm 2 — Yeni Araçlar

### 2.1 `tool_image_to_video`

Görüntüyü ilk kare olarak kullanarak video üretir. `last_frame_path` verilirse first+last frame modu etkinleşir.

**Parametreler:**
```
first_frame_path  str       (zorunlu)
last_frame_path   str       (opsiyonel)
prompt            str       (hareket/sahne yönlendirme)
output_filename   str       (zorunlu, output_path ile birlikte opsiyonel olabilir)
output_path       str       (opsiyonel, tam yol)
duration          int       4 / 6 / 8
aspect_ratio      str       "16:9" / "9:16"
model_tier        str       fast / quality
```

### 2.2 `tool_extend_video`

Var olan bir videoyu uzatır.

**Parametreler:**
```
video_path      str    (zorunlu)
output_filename str
output_path     str    (opsiyonel)
prompt          str    (opsiyonel, uzatma yönlendirme)
extra_seconds   int    4 / 6 / 8
model_tier      str    fast / quality
```

### 2.3 `tool_video_object_edit`

Video içine nesne ekler veya kaldırır.

**Parametreler:**
```
video_path      str
operation       str    "insert" / "remove"
prompt          str    (ne ekleneceği/kaldırılacağı)
output_filename str
output_path     str    (opsiyonel)
model_tier      str    fast / quality
```

### 2.4 `tool_upload_file`

Gemini Files API'ye dosya yükler. Dönen `file_uri` diğer araçlarda (örn. `transform_image`'de `additional_image_paths`) referans olarak kullanılabilir.

**Parametreler:**
```
file_path     str    (zorunlu)
mime_type     str    (opsiyonel, auto-detect)
display_name  str    (opsiyonel)
```

**Dönüş:**
```json
{ "file_uri": "...", "name": "...", "mime_type": "...", "size_bytes": 0, "expires_at": "..." }
```

### 2.5 `tool_batch_generate`

Birden fazla prompt'u tek MCP çağrısında paralel üretir (`asyncio.gather`, max 4 concurrent).

**Parametreler:**
```
prompts         List[str]   (max 10 prompt)
output_prefix   str         (örn. "batch_item" → batch_item_0.png, batch_item_1.png)
output_dir      str         (opsiyonel, tam dizin yolu; yoksa DEFAULT_OUTPUT_DIR)
model_tier      str         fast / balanced / quality / ultra
aspect_ratio    str         (default "1:1")
```

**Dönüş:** Her prompt için ayrı sonuç nesnesi; başarısız olanlar `success: false` ile döner, diğerleri etkilenmez.

### 2.6 `tool_run_pipeline`

Sık kullanılan işlem zincirlerini tek çağrıda çalıştırır. Her adımın çıktısı sonrakine `base_image_path` olarak otomatik bağlanır.

**Parametreler:**
```
steps           List[PipelineStep]
output_path     str    (opsiyonel, son adımın çıktısı için tam yol)
```

`PipelineStep`:
```
tool    "generate" | "edit" | "upscale" | "remove_background" | "transform"
params  dict   (o araca özgü parametreler; output dosyası pipeline tarafından yönetilir)
```

Örnek kullanım:
```json
{
  "steps": [
    {"tool": "generate", "params": {"prompt": "red candy", "model_tier": "quality"}},
    {"tool": "remove_background", "params": {}},
    {"tool": "upscale", "params": {}}
  ],
  "output_path": "C:/outputs/candy_final.png"
}
```

Hata davranışı: Bir adım başarısız olursa pipeline durur; o ana kadarki sonuçlar ve hata mesajı döner. Geçici dosyalar `temp/pipeline_<uuid>/` altında oluşturulur ve pipeline bitiminde temizlenir. Pipeline, ara adımlara otomatik dosya adı atar (`pipeline_<uuid>_step_0.png` vb.); kullanıcı yalnızca son adımın çıktısını `output_path` ile belirler.

### 2.7 `tool_generate_music`

Lyria 2 / Lyria 3 ile text-to-music üretimi.

**Parametreler:**
```
prompt          str    (zorunlu)
output_filename str    (örn. "track.mp3")
output_path     str    (opsiyonel)
model_name      str    "lyria-2" / "lyria-3"  (default: "lyria-2")
duration        int    (saniye, opsiyonel)
```

> **Not:** Lyria API'sinin Vertex AI üzerindeki erişilebilirliği proje bazlı değişebilir. İlk implementasyonda `tool_list_available_models` çıktısına dayalı runtime probe eklenir; erişilemeyen projeler için açık hata mesajı döner.

---

## Bölüm 3 — Mimari ve Kod Yapısı

### 3.1 Dosya Değişiklikleri

```
OpenGoogleImageGeneratorMCP/
├── mcp_server.py          ← Mevcut — yeni Params sınıfları + tool kayıtları eklenir
├── vertex_ai_tools.py     ← Mevcut — yeni core fonksiyonlar eklenir
├── config.py              ← Mevcut — değişmez
├── discovery.py           ← Mevcut — değişmez
├── model_registry.py      ← YENİ — tier → (model_name, api_backend) çözümü
└── pipeline.py            ← YENİ — adım zinciri motoru
```

### 3.2 `model_registry.py`

```python
TIER_MAP = {
    "generate":  { "fast": (...), "balanced": (...), "quality": (...), "ultra": (...) },
    "transform": { "fast": (...), "quality": (...) },
    "video":     { "fast": (...), "quality": (...) },
}

def resolve_model(tier: str, tool_type: str) -> tuple[str, str]:
    """Returns (model_name, api_backend). Falls back to 'fast' if tier unknown."""
```

### 3.3 `pipeline.py`

Sorumlulukları:
- Adımları sıralı çalıştır (paralel değil — çıktı bağımlılığı var)
- Adım N çıktısını adım N+1'e `base_image_path` olarak aktar
- Geçici dosya yaşam döngüsünü yönet (`temp/pipeline_<uuid>/`)
- Hata durumunda kısmi sonuçları döndür

### 3.4 `vertex_ai_tools.py` Eklenen Fonksiyonlar

```python
async def image_to_video(first_frame_path, output_path, prompt, ...) -> dict
async def extend_video(video_path, output_path, prompt, extra_seconds, ...) -> dict
async def video_object_edit(video_path, operation, prompt, output_path, ...) -> dict
async def upload_file(file_path, mime_type, display_name) -> dict
async def batch_generate(prompts, output_prefix, output_dir, ...) -> list[dict]
async def generate_music(prompt, output_path, model_name, duration) -> dict
```

`batch_generate` içinde `asyncio.gather` kullanılır; Vertex AI rate limit gözetilerek max 4 eşzamanlı istek.

### 3.5 Hata Yönetimi

Mevcut pattern korunur: tüm araçlar `{"success": bool, "error": str | None, ...}` döner. Pipeline için `steps[i].success` ayrı takip edilir.

### 3.6 Test Yaklaşımı

`tests/` altına eklenenler:
- `test_model_registry.py` — tier mapping unit testleri
- `test_pipeline.py` — mock ile adım zinciri testleri
- `test_vertex_ai_tools.py` — yeni fonksiyon testleri için ek case'ler

---

## Özet Tablo

| Değişiklik | Tür | Etkilenen Dosya |
|---|---|---|
| `model_tier` parametresi | Parametre | `mcp_server.py`, `model_registry.py` |
| `output_path` tam yol desteği | Parametre | `mcp_server.py`, `vertex_ai_tools.py` |
| `seed`, `negative_prompt`, `enhance_prompt` vb. | Parametre | `mcp_server.py`, `vertex_ai_tools.py` |
| `generate_video` genişlemesi | Parametre | `mcp_server.py`, `vertex_ai_tools.py` |
| `tool_image_to_video` | Yeni araç | `mcp_server.py`, `vertex_ai_tools.py` |
| `tool_extend_video` | Yeni araç | `mcp_server.py`, `vertex_ai_tools.py` |
| `tool_video_object_edit` | Yeni araç | `mcp_server.py`, `vertex_ai_tools.py` |
| `tool_upload_file` | Yeni araç | `mcp_server.py`, `vertex_ai_tools.py` |
| `tool_batch_generate` | Yeni araç | `mcp_server.py`, `vertex_ai_tools.py` |
| `tool_run_pipeline` | Yeni araç | `mcp_server.py`, `pipeline.py` |
| `tool_generate_music` | Yeni araç | `mcp_server.py`, `vertex_ai_tools.py` |
| `model_registry.py` | Yeni dosya | — |
| `pipeline.py` | Yeni dosya | — |
