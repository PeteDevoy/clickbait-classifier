import sys
import joblib
import requests
import urllib.parse
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model and feature vectorizer
model = joblib.load('clickbait_model.joblib')
vectorizer = joblib.load('count_vectorizer.joblib')

def fetch_title_if_youtube(user_input: str) -> str:
    """
    If user_input is a YouTube URL, fetch the title via YouTube's oEmbed endpoint.
    Otherwise, return user_input as-is.
    """
    try:
        parsed_url = urllib.parse.urlparse(user_input)

        # If there's no scheme or netloc, it's probably not a valid URL (just text).
        if not parsed_url.scheme or not parsed_url.netloc:
            return user_input

        # Check if the hostname is one of the known YouTube hosts
        hostname = parsed_url.hostname.lower()
        if hostname in ["www.youtube.com", "youtube.com", "youtu.be"]:
            # Encode the full URL so it can be safely passed to the oEmbed endpoint
            encoded_url = urllib.parse.quote(user_input, safe='')
            oembed_url = f"https://www.youtube.com/oembed?url={encoded_url}&format=json"

            response = requests.get(oembed_url)
            if response.status_code == 200:
                data = response.json()
                # Return the extracted title if it exists
                return data.get("title", user_input)
            else:
                return user_input
        else:
            return user_input
    except Exception:
        return user_input

@app.route('/', methods=['GET'])
def home():
    user_input = request.args.get('headline', '')
    title = ""
    is_clickbait = None

    if user_input:
        # Attempt to fetch title from YouTube if it's a YouTube URL
        title = fetch_title_if_youtube(user_input)

        # Vectorize and classify
        title_vectorized = vectorizer.transform([title])
        prediction = model.predict(title_vectorized)[0]
        is_clickbait = int(prediction)

    return render_template("home.html",
                           title=title,
                           is_clickbait=is_clickbait)

def handle_cli():
    # Read the headline from the command-line argument
    user_input = sys.argv[1]

    # Attempt to fetch YouTube title if URL, otherwise return raw input
    title = fetch_title_if_youtube(user_input)

    # Vectorize the title using the trained feature vectorizer
    title_vectorized = vectorizer.transform([title])

    # Make a prediction using the trained model
    prediction = model.predict(title_vectorized)[0]

    # Output the prediction (0 for non-clickbait, 1 for clickbait)
    print(prediction)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run in CLI mode
        handle_cli()
    else:
        # Run the Flask development server
        app.run(debug=True)
