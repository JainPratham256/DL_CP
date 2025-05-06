import streamlit as st
import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import numpy as np
import time
import os

# Set page configuration
st.set_page_config(
    page_title="Music Genre Studio", 
    page_icon="🎵", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Define colors and theme
primary_color = "#6a1b9a"
secondary_color = "#9c27b0"
accent_color = "#e040fb"
bg_color = "#121212"
text_color = "#ffffff"
highlight_color = "#ffeb3b"

# Use inline SVG for animations
music_icon_svg = """
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="#e040fb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M9 18V5l12-2v13"></path>
    <circle cx="6" cy="18" r="3"></circle>
    <circle cx="18" cy="16" r="3"></circle>
</svg>
"""

# Genre icons and visuals
genre_icons = {
    "blues": "🎷",
    "classical": "🎻",
    "country": "🤠",
    "disco": "🪩",
    "hiphop": "🎤",
    "jazz": "🎺",
    "metal": "🤘",
    "pop": "🎵",
    "reggae": "🎸",
    "rock": "🔥"
}

genre_colors = {
    "blues": "#0d47a1",
    "classical": "#880e4f",
    "country": "#e65100",
    "disco": "#6200ea",
    "hiphop": "#424242",
    "jazz": "#00838f",
    "metal": "#bf360c",
    "pop": "#d500f9",
    "reggae": "#33691e",
    "rock": "#c62828"
}

# Custom CSS for music studio look
st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(135deg, {bg_color} 0%, #1d1d1d 100%);
        color: {text_color};
        font-family: 'Helvetica Neue', sans-serif;
    }}
    .stButton>button {{
        background: linear-gradient(90deg, {primary_color} 0%, {secondary_color} 100%);
        color: white;
        border-radius: 24px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 12px rgba(106, 27, 154, 0.4);
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(106, 27, 154, 0.6);
        background: linear-gradient(90deg, {secondary_color} 0%, {accent_color} 100%);
    }}
    [data-testid="stFileUploader"] {{
        background: rgba(45, 45, 45, 0.6);
        padding: 25px;
        border-radius: 16px;
        border: 2px dashed {accent_color};
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: {highlight_color};
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.4);
    }}
    .title-container {{
        text-align: center;
        margin-bottom: 20px;
        animation: glow 1.5s ease-in-out infinite alternate;
    }}
    .title {{
        font-size: 4em;
        font-weight: 800;
        background: linear-gradient(45deg, {accent_color}, {highlight_color});
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin-bottom: 10px;
        font-family: 'Helvetica Neue', sans-serif;
    }}
    .subtitle {{
        font-size: 1.5em;
        color: {text_color};
        opacity: 0.8;
        font-weight: 300;
    }}
    .result-container {{
        background: rgba(30, 30, 30, 0.7);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        text-align: center;
        border-left: 5px solid {accent_color};
        margin: 20px 0;
    }}
    .genre-title {{
        font-size: 3em;
        font-weight: 700;
        margin-bottom: 15px;
        text-shadow: 0 0 10px rgba(224, 64, 251, 0.5);
    }}
    .genre-icon {{
        font-size: 5em;
        margin: 10px;
        animation: pulse 2s infinite;
    }}
    .audio-container {{
        background: rgba(20, 20, 20, 0.6);
        border-radius: 16px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        border-left: 5px solid {primary_color};
    }}
    .audio-title {{
        font-size: 1.5em;
        color: {highlight_color};
        margin-bottom: 15px;
        font-weight: 500;
    }}
    .stAudio {{
        width: 100%;
        border-radius: 10px;
        background: rgba(40, 40, 40, 0.7) !important;
    }}
    .info-box {{
        background: rgba(35, 35, 35, 0.7);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        border-left: 3px solid {highlight_color};
    }}
    .loading {{
        font-size: 1.2em;
        color: {highlight_color};
        text-align: center;
        margin: 20px 0;
        animation: pulse 1.5s infinite;
    }}
    @keyframes glow {{
        from {{
            text-shadow: 0 0 5px {accent_color}, 0 0 10px {accent_color}, 0 0 15px {secondary_color};
        }}
        to {{
            text-shadow: 0 0 10px {accent_color}, 0 0 20px {accent_color}, 0 0 30px {secondary_color};
        }}
    }}
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
        100% {{ transform: scale(1); }}
    }}
    .stProgress > div > div {{
        background-color: {accent_color};
    }}
    .genre-meter {{
        height: 20px;
        background: rgba(40, 40, 40, 0.6);
        border-radius: 10px;
        margin: 5px 0;
        position: relative;
        overflow: hidden;
    }}
    .genre-meter-fill {{
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-out;
    }}
    .genre-meter-text {{
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        display: flex;
        align-items: center;
        padding-left: 10px;
        font-weight: 500;
        color: white;
    }}
    .stSlider {{
        padding-top: 2rem;
    }}
    .stSlider > div > div > div {{
        background-color: {accent_color} !important;
    }}
    .equalizer {{
        display: flex;
        justify-content: center;
        align-items: flex-end;
        height: 60px;
        margin: 20px 0;
    }}
    .equalizer-bar {{
        width: 5px;
        background: {accent_color};
        margin: 0 3px;
        border-radius: 5px;
        animation: equalize 1s infinite;
    }}
    .equalizer-bar:nth-child(1) {{ animation-delay: 0.0s; }}
    .equalizer-bar:nth-child(2) {{ animation-delay: 0.1s; }}
    .equalizer-bar:nth-child(3) {{ animation-delay: 0.2s; }}
    .equalizer-bar:nth-child(4) {{ animation-delay: 0.3s; }}
    .equalizer-bar:nth-child(5) {{ animation-delay: 0.4s; }}
    .equalizer-bar:nth-child(6) {{ animation-delay: 0.1s; }}
    .equalizer-bar:nth-child(7) {{ animation-delay: 0.2s; }}
    .equalizer-bar:nth-child(8) {{ animation-delay: 0.3s; }}
    .equalizer-bar:nth-child(9) {{ animation-delay: 0.4s; }}
    .equalizer-bar:nth-child(10) {{ animation-delay: 0.1s; }}
    .equalizer-bar:nth-child(11) {{ animation-delay: 0.2s; }}
    .equalizer-bar:nth-child(12) {{ animation-delay: 0.3s; }}
    @keyframes equalize {{
        0% {{ height: 10px; }}
        50% {{ height: 40px; }}
        100% {{ height: 10px; }}
    }}
    .music-icon {{
        display: flex;
        justify-content: center;
        animation: pulse 2s infinite;
    }}
    .vinyl {{
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, #000 30%, transparent 30%), 
                    repeating-radial-gradient(circle, #333 5%, #222 10%);
        border-radius: 50%;
        box-shadow: 0 0 20px rgba(0,0,0,0.6);
        margin: 20px auto;
        position: relative;
        animation: rotate 4s linear infinite;
    }}
    .vinyl::after {{
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 40px;
        height: 40px;
        background: {highlight_color};
        border-radius: 50%;
        transform: translate(-50%, -50%);
    }}
    @keyframes rotate {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    .wave {{
        height: 60px;
        width: 100%;
        position: relative;
        margin-bottom: 20px;
        background: linear-gradient(to bottom, rgba(106, 27, 154, 0.1) 0%, rgba(106, 27, 154, 0.3) 100%);
        overflow: hidden;
        border-radius: 10px;
    }}
    .wave::before {{
        content: "";
        position: absolute;
        left: 0;
        right: 0;
        height: 20px;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 320'%3E%3Cpath fill='%236a1b9a' fill-opacity='0.5' d='M0,224L48,213.3C96,203,192,181,288,181.3C384,181,480,203,576,202.7C672,203,768,181,864,181.3C960,181,1056,203,1152,202.7C1248,203,1344,181,1392,170.7L1440,160L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z'%3E%3C/path%3E%3C/svg%3E");
        background-size: 1440px 100%;
        animation: wave 10s infinite linear;
    }}
    @keyframes wave {{
        0% {{ background-position: 0 0; }}
        100% {{ background-position: 1440px 0; }}
    }}
    </style>
""", unsafe_allow_html=True)

# Create main layout with animations
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    # Title with animation
    st.markdown('<div class="title-container">'
                '<h1 class="title">🎵 MUSIC GENRE STUDIO</h1>'
                '<p class="subtitle">Discover the genre of any track with AI</p>'
                '</div>', unsafe_allow_html=True)
    
    # Custom music icon SVG
    st.markdown(f'<div class="music-icon">{music_icon_svg}</div>', unsafe_allow_html=True)
    
    # Add equalizer animation
    equalizer_html = '<div class="equalizer">'
    for i in range(12):
        equalizer_html += f'<div class="equalizer-bar" style="height: {10 + (i % 4) * 10}px;"></div>'
    equalizer_html += '</div>'
    
    st.markdown(equalizer_html, unsafe_allow_html=True)

# Load model and feature extractor
@st.cache_resource
def load_model():
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "leo-kwan/wav2vec2-base-100k-gtzan-music-genres-finetuned-gtzan"
    )
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "leo-kwan/wav2vec2-base-100k-gtzan-music-genres-finetuned-gtzan"
    )
    return model.eval(), extractor

# Create tabs for different sections
tab1, tab2 = st.tabs(["🎵 Classify Music", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader
        uploaded_file = st.file_uploader("Upload a .wav audio file to classify its music genre", 
                                        type=["wav"],
                                        help="For best results, upload a 3-10 second clip")
        
        if uploaded_file is not None:
            # Show uploaded audio in styled container
            st.markdown('<div class="audio-container">'
                       '<div class="audio-title">Your Track</div>', 
                       unsafe_allow_html=True)
            st.audio(uploaded_file, format="audio/wav")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Analyze button with loading animation
            analyze_button = st.button("🔍 Analyze Genre")
            
            if analyze_button:
                try:
                    # Show loading animation
                    st.markdown('<div class="loading">Analyzing audio patterns...</div>', 
                               unsafe_allow_html=True)
                    analysis_placeholder = st.empty()
                    
                    # Fancy progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        # Update progress bar
                        progress_bar.progress(i + 1)
                        time.sleep(0.02)
                    
                    # Analysis animation
                    with analysis_placeholder.container():
                        st.markdown(
                            '<div class="equalizer" style="height:150px">' + 
                            ''.join(['<div class="equalizer-bar" style="height:30px"></div>' for _ in range(12)]) +
                            '</div>', 
                            unsafe_allow_html=True
                        )
                    
                    # Load model on demand
                    model, extractor = load_model()
                    
                    # Process the audio file
                    waveform, sample_rate = torchaudio.load(uploaded_file)
                    waveform = waveform.mean(dim=0).unsqueeze(0)  # Convert to mono
                    
                    # Resample if the sample rate is not 16000
                    if sample_rate != 16000:
                        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                        waveform = resampler(waveform)
                    
                    # Feature extraction
                    inputs = extractor(waveform.numpy()[0], sampling_rate=16000, return_tensors="pt", padding=True)
                    
                    # Make predictions
                    with torch.no_grad():
                        logits = model(**inputs).logits
                        predicted_id = torch.argmax(logits, dim=-1).item()
                        
                        # Get softmax probabilities for all genres
                        probs = torch.nn.functional.softmax(logits, dim=-1)[0].numpy()
                    
                    # Get the predicted genre
                    genre = model.config.id2label[predicted_id]
                    
                    # Clear the loading animations
                    analysis_placeholder.empty()
                    
                    # Display the result with animation
                    st.markdown(f"""
                        <div class="result-container" style="border-left: 5px solid {genre_colors.get(genre, accent_color)};">
                            <div class="genre-icon">{genre_icons.get(genre, '🎵')}</div>
                            <div class="genre-title" style="color: {genre_colors.get(genre, accent_color)};">
                                {genre.upper()}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
      
                    
                  
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please try another audio file or ensure it's a valid .wav format.")
    
    with col2:
        # Add vinyl animation
        st.markdown('<div class="vinyl"></div>', unsafe_allow_html=True)
        
        # Add wave animation
        st.markdown('<div class="wave"></div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h3>Tips for Best Results</h3>
            <ul>
                <li>Use high-quality WAV files</li>
                <li>Clips of 3-10 seconds work best</li>
                <li>Choose segments with clear instrumentation</li>
                <li>Avoid clips with mixed genres</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h3>Supported Genres</h3>
            <ul>
                <li>🎷 Blues</li>
                <li>🎻 Classical</li>
                <li>🤠 Country</li>
                <li>🪩 Disco</li>
                <li>🎤 Hip Hop</li>
                <li>🎺 Jazz</li>
                <li>🤘 Metal</li>
                <li>🎵 Pop</li>
                <li>🎸 Reggae</li>
                <li>🔥 Rock</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ## About the Classifier
        
        This music genre classifier uses a fine-tuned Wav2Vec2 model to identify different genres of music from audio clips. The model has been trained on the GTZAN dataset, which contains 1000 audio tracks, each 30 seconds long, with 100 examples per genre across 10 genres.
        
        ### How It Works
        
        1. Your audio is converted to a suitable format
        2. The Wav2Vec2 model extracts audio features
        3. The classifier analyzes these features
        4. The genre with highest probability is displayed
        
        ### Limitations
        
        - Works best with clear, single-genre audio clips
        - Some genres may be more difficult to classify than others
        - Very short clips might not provide enough data for accurate classification
        """)
    
    with col2:
        st.markdown("""
        ## Technology Used
        
        This application is built with:
        
        - **Streamlit**: For the web interface
        - **Wav2Vec2**: A state-of-the-art model for audio processing
        - **PyTorch & Torchaudio**: For audio manipulation
        - **Hugging Face Transformers**: For accessing pre-trained models
        
        ### Future Improvements
        
        - Multi-genre detection for mixed music
        - Support for more audio formats
        - Detailed musical feature analysis
        - Recommendations based on the detected genre
        """)

# Add footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; opacity: 0.7;">
    <p>Created with ❤️ for music enthusiasts</p>
</div>
""", unsafe_allow_html=True)

# Add custom CSS for a music visualizer effect at the bottom
st.markdown("""
<div class="visualizer-container">
    <div class="visualizer-bar" style="height: 30px; background: linear-gradient(90deg, #6a1b9a, #9c27b0, #e040fb, #f50057, #ff4081); border-radius: 10px; margin-top: 20px; animation: pulse 2s infinite;"></div>
</div>
""", unsafe_allow_html=True)
