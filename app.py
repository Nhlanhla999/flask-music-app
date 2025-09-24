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


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        files = request.files.getlist('music_files')
        if not files:
            return "No files uploaded", 400

        # Create a unique folder to store this upload
        folder_id = str(uuid.uuid4())
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_id)
        os.makedirs(folder_path, exist_ok=True)

        # Save each file directly in the folder (flatten structure)
        for file in files:
            filename_only = os.path.basename(file.filename)  # remove any subfolder path
            save_path = os.path.join(folder_path, filename_only)
            file.save(save_path)

        # Initialize playlist
        playlist = {mood: [] for mood in MOODS}
        playlist["mp4"] = {}

        # Process MP3 and MP4 files
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            if file.endswith('.mp3'):
                mood = classify_song(file_path)
                if mood in playlist:
                    playlist[mood].append(file)  # just filename
            elif file.endswith('.mp4'):
                features = get_audio_features(file_path, mode="similarity")
                if features.size > 0:
                    audio_features_cache[file_path] = features

        save_features_to_disk()
        precompute_folder_features(folder_path)

        # Compute top similar MP4s
        mp4_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".mp4")]
        for file in mp4_files:
            file_path = os.path.join(folder_path, file)
            target_feats = audio_features_cache.get(file_path)
            if target_feats is None:
                continue
            sims = [
                (os.path.basename(other_file), cosine_similarity(target_feats, feats))
                for other_file, feats in audio_features_cache.items()
                if other_file != file_path
            ]
            sims.sort(key=lambda x: x[1], reverse=True)
            playlist["mp4"][file] = [f for f, _ in sims[:10]]

        # Save playlist JSON
        with open(os.path.join(folder_path, 'playlist.json'), 'w') as f:
            json.dump(playlist, f, indent=2)

        # Initialize liked/disliked JSON
        liked_path = os.path.join(folder_path, 'liked.json')
        disliked_path = os.path.join(folder_path, 'disliked.json')
        if not os.path.exists(liked_path):
            save_json(liked_path, {"liked": []})
        if not os.path.exists(disliked_path):
            save_json(disliked_path, {"disliked": []})

        return redirect(url_for('player', folder_id=folder_id))

    return render_template("upload.html")

@app.route('/', methods=['GET', 'POST'])
def index():
    load_features_cache()
    uploads = os.listdir(app.config['UPLOAD_FOLDER'])
    folder_id = None
    playlist = {}
    liked_songs = []
    disliked_songs = []

    # Look for the most recent folder with playlist.json
    for folder in sorted(uploads, reverse=True):
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
        playlist_path = os.path.join(folder_path, 'playlist.json')
        if os.path.isdir(folder_path) and os.path.exists(playlist_path):
            folder_id = folder
            with open(playlist_path, 'r') as f:
                playlist = json.load(f)

            liked_path = os.path.join(folder_path, 'liked.json')
            disliked_path = os.path.join(folder_path, 'disliked.json')
            if not os.path.exists(liked_path):
                save_json(liked_path, {"liked": []})
            if not os.path.exists(disliked_path):
                save_json(disliked_path, {"disliked": []})

            liked_songs = load_json(liked_path, {"liked": []})["liked"]
            disliked_songs = load_json(disliked_path, {"disliked": []})["disliked"]
            break

    if not folder_id:
        return render_template(
            'home.html',
            message="No playlist found. Please upload a music folder.",
            playlist={},
            folder_id=None,
            liked=[],
            disliked=[]
        )

    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_id)
    return render_template(
        'home.html',
        playlist=playlist,
        folder_id=folder_id,
        liked=liked_songs,
        disliked=disliked_songs
    )


@app.route('/player/<folder_id>', methods=['GET', 'POST'])
def player(folder_id):
    load_features_cache()
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_id)

    if not os.path.exists(folder_path):
        return f"Folder {folder_id} not found.", 404

    # Load existing playlist
    playlist_path = os.path.join(folder_path, 'playlist.json')
    if os.path.exists(playlist_path):
        with open(playlist_path) as f:
            playlist = json.load(f)
    else:
        playlist = {mood: [] for mood in MOODS}
        playlist["mp4"] = {}

    liked_data = load_json(os.path.join(folder_path, 'liked.json'), {"liked": []})
    disliked_data = load_json(os.path.join(folder_path, 'disliked.json'), {"disliked": []})

    return render_template(
        'home.html',
        playlist=playlist,
        folder_id=folder_id,
        liked=liked_data["liked"],
        disliked=disliked_data["disliked"]
    )



@app.route('/library/<folder_id>')
def library(folder_id):
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_id)

    # Load main playlist
    with open(os.path.join(folder_path, 'playlist.json')) as f:
        playlist = json.load(f)

    # Flatten all songs
    all_songs = []
    for mood, songs in playlist.items():
        for song in songs:
            all_songs.append({'title': song, 'mood': mood})

    # Support both string and dict in case songs include metadata like duration
    mp3_songs = [s for s in all_songs if (s['title'] if isinstance(s, dict) else s).lower().endswith('.mp3')]
    mp4_songs = [s for s in all_songs if (s['title'] if isinstance(s, dict) else s).lower().endswith('.mp4')]

    liked_path = os.path.join(folder_path, 'liked.json')
    liked_data = load_json(liked_path, {"liked": []})
    liked_songs = liked_data["liked"]

    return render_template(
        'home.html',
        songs=all_songs,
        liked_songs=liked_songs,
         mp3_songs=mp3_songs,
        mp4_songs=mp4_songs,
        folder_id=folder_id
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

@app.route('/songs/<folder_id>')
def get_songs_by_mood(folder_id):
    mood = request.args.get('mood', 'all')
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_id)

    with open(os.path.join(folder_path, 'playlist.json')) as f:
        playlist = json.load(f)

    if mood == 'all':
        songs = [
            {'title': song, 'mood': m}
            for m, songlist in playlist.items()
            for song in songlist if not song.lower().endswith('.mp4')  # ⛔ Exclude .mp4
        ]
    else:
        songs = [
            {'title': song, 'mood': mood}
            for song in playlist.get(mood, [])
            if not song.lower().endswith('.mp4')  # ⛔ Exclude .mp4
        ]

    return jsonify({'songs': songs})  # ✅ Wrap the response


@app.route('/api/playlist/<folder_id>')
def get_playlist(folder_id):
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_id)
    playlist_path = os.path.join(folder_path, 'playlist.json')

    if os.path.exists(playlist_path):
        with open(playlist_path, 'r') as f:
            playlist = json.load(f)
        return jsonify(playlist)

    return jsonify({'error': 'Playlist not found'}), 404

@app.route('/api/home')
def api_home():
    uploads = os.listdir(app.config['UPLOAD_FOLDER'])
    folder_id = None
    playlist = {}

    for folder in sorted(uploads, reverse=True):
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
        playlist_path = os.path.join(folder_path, 'playlist.json')
        if os.path.isdir(folder_path) and os.path.exists(playlist_path):
            folder_id = folder
            with open(playlist_path, 'r') as f:
                playlist = json.load(f)
            break

    return render_template('partials/home_content.html', playlist=playlist)


# ----------------------------
# API: Load partial content for Library
# ----------------------------
@app.route('/api/library/<folder_id>')
def api_library(folder_id):
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_id)
    playlist_path = os.path.join(folder_path, 'playlist.json')

    if not os.path.exists(playlist_path):
        return jsonify({'error': 'Playlist not found'}), 404

    with open(playlist_path, 'r') as f:
        playlist = json.load(f)

    # Flatten songs for frontend
    all_songs = []
    for mood, songs in playlist.items():
        for song in songs:
            all_songs.append({'title': song, 'mood': mood})

    # Support both string and dict in case songs include metadata like duration
    mp3_songs = [s for s in all_songs if (s['title'] if isinstance(s, dict) else s).lower().endswith('.mp3')]
    mp4_songs = [s for s in all_songs if (s['title'] if isinstance(s, dict) else s).lower().endswith('.mp4')]
    liked_path = os.path.join(folder_path, 'liked.json')
    liked_data = load_json(liked_path, {"liked": []})

    return render_template(
        'partials/library_content.html',
        songs=all_songs,
        liked_songs=liked_data["liked"],
        folder_id=folder_id,
        mp3_songs=mp3_songs,
        mp4_songs=mp4_songs
    )
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


