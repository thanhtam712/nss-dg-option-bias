
# NSS: Generate textual data and demo samples (direction vectors)

Goal: create **demo samples** used to compute NSS directions (visual/textual) for `NSS_DG`.

## 1) Setup

- Install base deps: `pip install -r requirements.txt`
- If you generate textuals with Gemini: `pip install google-genai python-dotenv tqdm`
- API keys:
  - Gemini generation requires `GEMINI_API_KEY` in a repo-root `.env` file (loaded by `python-dotenv`).
  - `OPENAI_API_KEY` is only needed if you use the OpenAI caption scripts.

## 2) Using your pre-generated Google Drive data

I can’t fetch Google Drive links from this environment. Please download the folders locally, then place/unzip them into this repo.

Recommended target locations:

- Textual/Q&A JSON (has `caption_redirected`): `data/data_samples/<dataset>/baseline_gemini_combined.json`
- NSS demo samples (optional, if already created): `NSS_DG/experiments/data/<dataset>/gemini_combined.json`
- Images/keyframes:
  - image datasets: a single image folder (e.g. `.../val2017/*.jpg`)
  - video datasets: `images_dir/<video_id>/*.jpg`

Unzip everything (run from repo root):

```bash
# Example: you downloaded archives into ~/Downloads/nss_data
mkdir -p data NSS_DG/experiments/data

find ~/Downloads/nss_data -maxdepth 2 -type f \( -name '*.zip' -o -name '*.tar.gz' -o -name '*.tgz' \) -print

# unzip zip files
find ~/Downloads/nss_data -type f -name '*.zip' -exec unzip -o {} -d . \;

# extract tar.gz / tgz
find ~/Downloads/nss_data -type f \( -name '*.tar.gz' -o -name '*.tgz' \) -exec tar -xzf {} -C . \;
```

## 3) Generate textual data (field: `caption_redirected`)

NSS expects an “additional information” string in `caption_redirected`.

### Option A: already available

Use existing files like:

- `data/data_samples/<dataset>/baseline_gemini_combined.json`

Each item should include at least:

- `image_id` or `video_id`
- `question`
- `answer_choice_0..N`
- `correct_answer_id`, `outline_answer_id`
- `caption_redirected`

### Option B: generate with Gemini

Create a `.env` file at the repo root:

```bash
echo "GEMINI_API_KEY=YOUR_KEY" > .env
```

```bash
python src/preprocess/gemini_gen_textual.py \
  --images-root <images_or_keyframes_dir> \
  --question-path <input_questions_json> \
  --answer-path <output_json>
```

Note: this script writes `textual_*` fields (e.g. `textual_redirected`). If your pipeline needs `caption_redirected`, map:

- `caption_redirected <- textual_redirected`

## 4) Build NSS demo samples (for direction computation)

```bash
python src/preprocess/prepare_data_samples_nss.py \
  --data-path data/data_samples/<dataset>/baseline_gemini_combined.json \
  --images-dir <images_or_keyframes_dir> \
  --output-path NSS_DG/experiments/data/<dataset>/gemini_combined.json
```

The output demo sample schema:

- `id`, `images` (list of filenames)
- `question` (formatted prompt with options + additional info)
- `value` (correct answer, format: `"<id>: <answer>"`)
- `h_value` (redirect/hallucinated answer, same format)

## 5) Run NSS (example)

Pass the demo file via `--samples-file` and the image/keyframe folder via `--image-folder`.

```bash
python NSS_DG/experiments/eval/qwen.py \
  --samples-file NSS_DG/experiments/data/aok_vqa/gemini_combined.json \
  --image-folder <images_or_keyframes_dir> \
  --alpha_text 0.1 --alpha_image 0.0
```
