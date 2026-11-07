# EdgeDepth AI integration patch

Copy `app.py` into the project root (replace the placeholder Flask app).
Copy `requirements.txt` into the project root.

The existing `templates/index.html` and `static/js/script.js` expect:
- `GET /health`
- `POST /reconstruct` multipart fields: `rgb`, `depth`, optional `ground_truth`,
  `scale`, `method`, `dataset`, `noise_reduction`, `hole_filling`,
  `edge_preservation`, `rgb_guidance`, `boundary_refinement`.

The supplied app implements the API response fields used by the current JS:
`original`, `refined`, `edges`, `boundary`, `mae`, `rmse`, `psnr`, `ssim`,
`processing_time`, `output_resolution`, `memory_usage`.

Important:
1. Run from the project root so `backend/`, `classical/`, `models/`,
   `templates/`, and `static/` are importable/available.
2. Deep inference needs a checkpoint compatible with the model architecture
   and selected scale. Set `EDGEDepth_CHECKPOINT` to its path. The default is
   `outputs/training/nyu/best.pth`.
3. A NYU depth checkpoint predicts metric depth; a Middlebury checkpoint
   predicts disparity. Choose matching input units. The API does not convert
   between meters and disparity pixels.
4. The frontend's ground-truth upload must have the exact output resolution.
5. `joint_bilateral` uses OpenCV's ximgproc implementation if available;
   otherwise it falls back to a bilateral filter.
6. Your ZIP did not contain `backend/`, `classical/`, `models/`, or training
   code, so this patch assumes those folders remain in your working project.
7. Install dependencies in your active virtual environment:
   `python -m pip install -r requirements.txt`
8. Start:
   `python app.py`
