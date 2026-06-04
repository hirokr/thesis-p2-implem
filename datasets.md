Checked the three metadata files.

**Result**
- `av1.metadata.json`: **not balanced for real/fake**
  - Real: `1228` / `5000` = `24.56%`
  - Fake: `3772` / `5000` = `75.44%`
  - But its native 4 classes are nearly balanced:
    - `real`: `1228`
    - `visual_modified`: `1234`
    - `audio_modified`: `1238`
    - `both_modified`: `1300`

- `dfdc.metadata.json`: **approximately balanced**
  - Real: `2455` / `5000` = `49.10%`
  - Fake: `2545` / `5000` = `50.90%`

- `faceavceleb.metadata.json`: **not balanced for real/fake**
  - Real: `500` / `2000` = `25.00%`
  - Fake: `1500` / `2000` = `75.00%`
  - But its native 4 AV classes are perfectly balanced:
    - `RealVideo-RealAudio`: `500`
    - `RealVideo-FakeAudio`: `500`
    - `FakeVideo-RealAudio`: `500`
    - `FakeVideo-FakeAudio`: `500`

So: **DFDC is balanced for binary real/fake classification. AV1 and FakeAVCeleb are balanced by their 4-class audio/video manipulation categories, but not balanced if collapsed into real vs fake.**