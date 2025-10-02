from flask import Flask, request, redirect, url_for, render_template,jsonify,abort
import os, uuid, json
from mood_classifier import classify_song,get_audio_features,cosine_similarity,load_features_cache, precompute_folder_features,save_features_to_disk
import subprocess
from collections import Counter
import time
from glob import glob
import numpy as np
import difflib


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
audio_features_cache = {}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
FEATURES_FILE = "features_cache.json"
SIMILARITY_THRESHOLD = 0.7  # Only return videos with >= 70% similarity
MAX_RESULTS = 10
playlist_queue = {}


MOODS = ['feel_good', 'sad', 'energetic', 'relax','party', 'romance']

@app.route('/', methods=['GET', 'POST'])
def index():
 
    return render_template(
        'home.html'
       
    )

@app.route('/library')
def library():
  

    return render_template(
        'home.html',
      
    )


@app.route('/api/home')
def api_home():

    return render_template('partials/home_content.html')


@app.route('/api/library')
def api_library():
    return render_template(
        'partials/library_content.html'
     
    )



@app.route('/api/playlist/<folder_id>')
def api_playlist(folder_id):
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_id)
    playlist_path = os.path.join(folder_path, 'playlist.json')

    if not os.path.exists(playlist_path):
        return jsonify({"items": []})

    try:
        with open(playlist_path, 'r') as f:
            playlist = json.load(f)
    except Exception:
        return jsonify({"items": []})

    items = []
    for mood, songs in playlist.items():
        for s in songs:
            # support both simple string entries and dict metadata entries
            if isinstance(s, dict):
                title = s.get('title') or s.get('name') or ''
                meta = {k: v for k, v in s.items() if k not in ('title', 'name')}
            else:
                title = s
                meta = {}

            if not title:
                continue

            typ = 'mp4' if title.lower().endswith('.mp4') else 'mp3' if title.lower().endswith('.mp3') else 'other'
            items.append({
                "title": title,
                "mood": mood,
                "type": typ,
                "meta": meta
            })

    return jsonify({"items": items})


def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            # file is corrupted → reset
            return default
    return default

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

@app.route('/like/<folder_id>/<song>', methods=['POST'])
def like_song(folder_id, song):
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_id)
    liked_path = os.path.join(folder_path, 'liked.json')
    disliked_path = os.path.join(folder_path, 'disliked.json')

    liked_data = load_json(liked_path, {"liked": []})
    disliked_data = load_json(disliked_path, {"disliked": []})

    if song in liked_data["liked"]:
        liked_data["liked"].remove(song)
        status = "unliked"
    else:
        liked_data["liked"].append(song)
        if song in disliked_data["disliked"]:
            disliked_data["disliked"].remove(song)
        status = "liked"

    save_json(liked_path, liked_data)
    save_json(disliked_path, disliked_data)

    return jsonify({"status": status, "song": song})


@app.route('/dislike/<folder_id>/<song>', methods=['POST'])
def dislike_song(folder_id, song):
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_id)
    disliked_path = os.path.join(folder_path, 'disliked.json')
    liked_path = os.path.join(folder_path, 'liked.json')

    disliked_data = load_json(disliked_path, {"disliked": []})
    liked_data = load_json(liked_path, {"liked": []})

    if song in disliked_data["disliked"]:
        disliked_data["disliked"].remove(song)
        status = "undisliked"
    else:
        disliked_data["disliked"].append(song)
        if song in liked_data["liked"]:
            liked_data["liked"].remove(song)
        status = "disliked"

    save_json(liked_path, liked_data)
    save_json(disliked_path, disliked_data)

    return jsonify({"status": status, "song": song})



# JSON endpoints
@app.route('/liked/<folder_id>')
def get_liked_songs(folder_id):
    liked_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_id, 'liked.json')
    liked_data = load_json(liked_path, {"liked": []})
    return jsonify(liked_data["liked"])


@app.route('/disliked/<folder_id>')
def get_disliked_songs(folder_id):
    disliked_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_id, 'disliked.json')
    disliked_data = load_json(disliked_path, {"disliked": []})
    return jsonify(disliked_data["disliked"])

@app.route('/api/playlist/<folder_id>')
def get_playlist(folder_id):
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_id)
    playlist_path = os.path.join(folder_path, 'playlist.json')

    if os.path.exists(playlist_path):
        with open(playlist_path, 'r') as f:
            playlist = json.load(f)
        return jsonify(playlist)

    return jsonify({'error': 'Playlist not found'}), 404



def normalize_filename(name: str) -> str:
    """Lowercase, replace spaces with underscores, remove common URL encoding issues."""
    return (
        name.lower()
            .replace(" ", "_")
            .replace("%20", "_")
            .replace("#", "")
    )

@app.route('/similar_by_sound/<folder_id>/<filename>')
def similar_by_sound(folder_id, filename):
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_id)
    playlist_path = os.path.join(folder_path, "playlist.json")

    if not os.path.exists(playlist_path):
        return jsonify([])  # No playlist found

    playlist = load_json(playlist_path, {})
    mp4_playlist = playlist.get("mp4", {})

    if not mp4_playlist:
        return jsonify([])

    # Normalize requested filename
    norm_filename = normalize_filename(filename)

    # Normalize keys in playlist
    norm_keys = {normalize_filename(k): v for k, v in mp4_playlist.items()}

    # Try exact normalized match first
    similar_videos = norm_keys.get(norm_filename)

    # Fallback to fuzzy matching if exact match fails
    if similar_videos is None:
        closest = difflib.get_close_matches(norm_filename, norm_keys.keys(), n=1, cutoff=0.8)
        if closest:
            similar_videos = norm_keys[closest[0]]
        else:
            similar_videos = []

    # Return in expected format for JS
    response = [{"title": vid} for vid in similar_videos]
    return jsonify(response)



def log_mp4_play(folder_id, song):
    if not song.lower().endswith(".mp4"):
        return

    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_id)
    log_path = os.path.join(folder_path, "video_logs.json")

    data = load_json(log_path, {"history": []})
    data["history"].append({
        "song": os.path.basename(song).lower(),
        "timestamp": time.time()
    })
    save_json(log_path, data)


def get_recent_videos(folder_id, limit=10,exclude=None):
    """Return most recent mp4 plays"""
    exclude=set(exclude or [])
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_id)
    log_path = os.path.join(folder_path, "video_logs.json")
    data = load_json(log_path, {"history": []})

    # Sort by newest timestamp
    recent = sorted(data["history"], key=lambda x: x["timestamp"], reverse=True)
    filtered=[item["song"] for item in recent if item["song"] not in exclude]
    return filtered[:limit]


def get_frequent_videos(folder_id, limit=10,exclude=None):
    """Return most frequently played mp4"""
    exclude=set(exclude or [])
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_id)
    log_path = os.path.join(folder_path, "video_logs.json")
    data = load_json(log_path, {"history": []})

    counter = Counter([item["song"] for item in data["history"]])
    filtered=[song for song, _ in counter.most_common() if song not in exclude]
    return filtered[:limit]


@app.route('/log_play/<folder_id>/<song>', methods=['POST'])
def log_play(folder_id, song):
    log_mp4_play(folder_id, song)
    return jsonify({"status": "ok"})

@app.route('/recent_videos/<folder_id>')
def recent_videos(folder_id):
    exclude=request.args.get("exclude","").split(",") if request.args.get("exclude") else []
    return jsonify(get_recent_videos(folder_id, exclude=exclude))

@app.route('/frequent_videos/<folder_id>')
def frequent_videos(folder_id):
    exclude=request.args.get("exclude","").split(",") if request.args.get("exclude") else []
    return jsonify(get_frequent_videos(folder_id, exclude=exclude))



def repair_video_logs(upload_folder):
    fixed_files = []
    for log_path in glob(os.path.join(upload_folder, "**", "video_logs.json"), recursive=True):
        try:
            with open(log_path, "r") as f:
                content = f.read().strip()
            
            if not content:
                continue

            # Try normal JSON first
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "history" in data:
                    continue  # already valid
            except json.JSONDecodeError:
                pass

            # If multiple JSON objects exist, split and load them
            history = []
            decoder = json.JSONDecoder()
            idx = 0
            while idx < len(content):
                try:
                    obj, offset = decoder.raw_decode(content[idx:])
                    if "history" in obj and isinstance(obj["history"], list):
                        history.extend(obj["history"])
                    idx += offset
                except json.JSONDecodeError:
                    break  # stop on invalid data

            if history:
                repaired = {"history": history}
                with open(log_path, "w") as f:
                    json.dump(repaired, f, indent=2)
                fixed_files.append(log_path)

        except Exception as e:
            print(f"⚠️ Error repairing {log_path}: {e}")

    return fixed_files


# --------------------------
# NOW call it on startup
# --------------------------
with app.app_context():
    fixed = repair_video_logs(app.config['UPLOAD_FOLDER'])
    if fixed:
        print("✅ Repaired video_logs.json in:", fixed)
    else:
        print("ℹ️ No corrupted video_logs.json files found.")
