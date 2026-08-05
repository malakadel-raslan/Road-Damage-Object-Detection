# Road Damage Detection — Web App

A Gradio web app that lets users upload a road image and see detected damage
(potholes, cracks, etc.) using your YOLOv8 model.

## 1. Add your model weights

Copy your trained `best.pt` file (from the Kaggle notebook's
`experiments/.../weights/best.pt`) into this folder, next to `app.py`.

If you'd rather keep it somewhere else, set an environment variable instead:

```bash
export MODEL_PATH=/path/to/your/best.pt
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run locally

```bash
python app.py
```

Gradio will print a local URL (usually `http://127.0.0.1:7860`) — open it in
your browser. Gradio also gives you a temporary public link if you launch
with `demo.launch(share=True)`.

## 4. Deploy for real (options)

**A) Hugging Face Spaces (easiest, free)**
1. Create a new Space at huggingface.co/new-space, SDK = Gradio.
2. Upload `app.py`, `requirements.txt`, and your `best.pt` to the Space
   (drag-and-drop in the web UI, or `git push`).
3. The Space builds automatically and gives you a public URL.

**B) Your own server / cloud VM**
1. Copy this folder to the server.
2. `pip install -r requirements.txt`
3. Run with a process manager, e.g.:
   ```bash
   nohup python app.py &
   ```
   or behind `gunicorn`/`systemd` for production reliability.
4. Put it behind Nginx + HTTPS (e.g. via Let's Encrypt) if exposing publicly.

**C) Docker**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
```
Build and run:
```bash
docker build -t road-damage-app .
docker run -p 7860:7860 road-damage-app
```

## Notes

- The confidence slider lets users adjust the detection threshold live.
- `results.plot()` (from Ultralytics) draws the bounding boxes automatically.
- For larger traffic, consider serving the model behind a proper inference
  API (FastAPI + Uvicorn) instead of Gradio's built-in server, and put
  Gradio purely on the frontend calling that API.
