import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="DermaCare AI — Skin Disease Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "page" not in st.session_state:
    st.session_state.page = "Home"
if "model_running" not in st.session_state:
    st.session_state.model_running = False
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
.stApp { background: #f7f9fc; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.top-banner { background: #0a2d5e; color: rgba(255,255,255,0.75); text-align: center; padding: 8px 20px; font-size: 12px; letter-spacing: 0.03em; }
.top-banner strong { color: white; }
.hero-section { background: linear-gradient(160deg, #0a2d5e 0%, #1a3a70 55%, #163060 100%); border-radius: 16px; padding: 48px 40px; margin-bottom: 32px; }
.hero-badge { display: inline-flex; align-items: center; gap: 7px; background: rgba(14,159,142,0.15); border: 1px solid rgba(14,159,142,0.35); color: #4dd9c8; font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; padding: 5px 14px; border-radius: 20px; margin-bottom: 16px; }
.hero-title { font-family: 'Playfair Display', serif; font-size: 2.4rem; font-weight: 700; color: white; line-height: 1.2; margin-bottom: 12px; }
.hero-title span { color: #7dd3fc; }
.hero-desc { color: rgba(255,255,255,0.6); font-size: 15px; line-height: 1.75; font-weight: 300; max-width: 580px; margin-bottom: 28px; }
.hero-stats { display: flex; gap: 36px; }
.stat-num { font-family: 'Playfair Display', serif; font-size: 1.8rem; font-weight: 700; color: white; line-height: 1; }
.stat-label { font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 3px; }
.panel { background: white; border: 1px solid #dde5ef; border-radius: 12px; padding: 24px; box-shadow: 0 2px 8px rgba(15,40,80,0.06); margin-bottom: 18px; }
.panel-title { font-family: 'Playfair Display', serif; font-size: 1rem; font-weight: 600; color: #1a2332; margin-bottom: 4px; }
.panel-sub { font-size: 12px; color: #6b7a90; margin-bottom: 16px; }
.dx-banner { background: linear-gradient(135deg, #0a2d5e, #1a4a8a); border-radius: 12px; padding: 28px 32px; color: white; margin-bottom: 20px; position: relative; overflow: hidden; }
.dx-banner::after { content: "✚"; position: absolute; right: 24px; top: 50%; transform: translateY(-50%); font-size: 90px; color: rgba(255,255,255,0.04); }
.dx-label { font-size: 10px; letter-spacing: 0.2em; text-transform: uppercase; color: rgba(255,255,255,0.45); margin-bottom: 8px; }
.dx-name { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; color: white; margin-bottom: 16px; line-height: 1.2; }
.dx-pills { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; }
.pill { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600; padding: 5px 14px; border-radius: 20px; }
.pill-blue { background: rgba(255,255,255,0.12); color: white; }
.pill-green { background: rgba(22,163,74,0.2); color: #86efac; }
.pill-amber { background: rgba(217,119,6,0.2); color: #fcd34d; }
.pill-red { background: rgba(220,38,38,0.2); color: #fca5a5; }
.conf-bar-label { display: flex; justify-content: space-between; font-size: 11px; color: rgba(255,255,255,0.45); margin-bottom: 6px; }
.conf-bar-track { height: 5px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden; }
.conf-bar-fill { height: 100%; background: linear-gradient(to right, #4dd9c8, #60a5fa); border-radius: 3px; }
.pred-item { margin-bottom: 16px; }
.pred-row { display: flex; justify-content: space-between; margin-bottom: 5px; }
.pred-name { font-size: 13.5px; font-weight: 600; color: #1a2332; }
.pred-pct { font-size: 13px; color: #1a6bcc; font-weight: 500; }
.pred-bar-track { height: 6px; background: #eef2f7; border-radius: 3px; overflow: hidden; margin-bottom: 4px; }
.pred-bar-fill { height: 100%; border-radius: 3px; }
.pred-bar-1 { background: linear-gradient(to right, #1a6bcc, #0e9f8e); }
.pred-bar-2 { background: linear-gradient(to right, #60a5fa, #93c5fd); }
.pred-bar-3 { background: #dde5ef; }
.pred-desc { font-size: 12px; color: #6b7a90; line-height: 1.5; }
.chip-group { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { font-size: 12.5px; color: #1a4a8a; background: #eff6ff; border: 1px solid #dbeafe; padding: 5px 13px; border-radius: 20px; font-weight: 500; }
.treat-item { display: flex; gap: 13px; padding: 11px 0; border-bottom: 1px solid #eef2f7; }
.treat-item:last-child { border: none; padding-bottom: 0; }
.treat-num { width: 25px; height: 25px; background: #1a6bcc; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; flex-shrink: 0; margin-top: 2px; }
.treat-text { font-size: 13.5px; color: #3d4f66; line-height: 1.6; }
.sev-meter { display: flex; gap: 4px; margin: 10px 0 5px; }
.sev-block { flex: 1; height: 8px; border-radius: 2px; background: #eef2f7; }
.sev-green { background: #16a34a; }
.sev-amber { background: #d97706; }
.sev-red { background: #dc2626; }
.sev-labels { display: flex; justify-content: space-between; font-size: 10.5px; color: #9aaabb; }
.urgency-low { background: #f0fdf4; border: 1px solid #bbf7d0; color: #15803d; padding: 11px 16px; border-radius: 8px; font-size: 13px; font-weight: 500; }
.urgency-mid { background: #fffbeb; border: 1px solid #fde68a; color: #92400e; padding: 11px 16px; border-radius: 8px; font-size: 13px; font-weight: 500; }
.urgency-high { background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; padding: 11px 16px; border-radius: 8px; font-size: 13px; font-weight: 500; }
.disclaimer { background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 14px 16px; font-size: 12.5px; color: #78350f; line-height: 1.6; margin-top: 8px; }
.disclaimer strong { color: #92400e; display: block; margin-bottom: 3px; }
.low-conf-warn { background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 16px 18px; color: #991b1b; font-size: 13.5px; line-height: 1.6; }
.low-conf-warn strong { display: block; margin-bottom: 4px; font-size: 14px; }
.about-hero { background: linear-gradient(160deg, #0a2d5e 0%, #1a3a70 100%); border-radius: 16px; padding: 52px 40px; margin-bottom: 32px; text-align: center; }
.about-hero h1 { font-family: 'Playfair Display', serif; font-size: 2.5rem; color: white; margin-bottom: 12px; }
.about-hero p { color: rgba(255,255,255,0.6); font-size: 15px; line-height: 1.75; max-width: 620px; margin: 0 auto; }
.member-card { background: white; border: 1px solid #dde5ef; border-radius: 14px; padding: 28px 20px; text-align: center; box-shadow: 0 2px 8px rgba(15,40,80,0.06); }
.member-avatar { width: 64px; height: 64px; background: linear-gradient(135deg, #1a6bcc, #0e9f8e); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 26px; margin: 0 auto 14px; color: white; box-shadow: 0 3px 10px rgba(26,107,204,0.3); }
.member-name { font-family: 'Playfair Display', serif; font-size: 1rem; font-weight: 700; color: #1a2332; margin-bottom: 6px; }
.member-role { font-size: 12px; color: #6b7a90; margin-bottom: 10px; }
.member-tech { display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; }
.tech-tag { font-size: 11px; background: #eff6ff; color: #1a6bcc; border: 1px solid #dbeafe; padding: 3px 10px; border-radius: 20px; font-weight: 600; }
.tech-card { background: white; border: 1px solid #dde5ef; border-radius: 12px; padding: 22px; display: flex; align-items: flex-start; gap: 14px; box-shadow: 0 2px 6px rgba(15,40,80,0.05); margin-bottom: 14px; }
.tech-icon { font-size: 26px; flex-shrink: 0; }
.tech-name { font-weight: 700; font-size: 14px; color: #1a2332; margin-bottom: 4px; }
.tech-desc { font-size: 12.5px; color: #6b7a90; line-height: 1.6; }
.feat-grid { display: flex; gap: 16px; }
.feat-card { flex: 1; background: white; border: 1px solid #dde5ef; border-radius: 12px; padding: 22px 18px; }
.feat-icon { width: 44px; height: 44px; border-radius: 11px; display: flex; align-items: center; justify-content: center; font-size: 20px; margin-bottom: 12px; }
.feat-icon-blue { background: #eff6ff; }
.feat-icon-teal { background: #e6f7f5; }
.feat-icon-amber { background: #fffbeb; }
.feat-title { font-weight: 700; font-size: 14px; color: #1a2332; margin-bottom: 6px; }
.feat-desc { font-size: 12.5px; color: #6b7a90; line-height: 1.65; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = [
    "Eczema", "Melanoma", "Atopic Dermatitis", "Basal Cell Carcinoma",
    "Melanocytic Nevi", "Benign Keratosis", "Psoriasis",
    "Seborrheic Keratoses", "Tinea Ringworm", "Warts Molluscum"
]

DISEASE_INFO = {
    "Eczema": {"description": "A chronic condition causing dry, itchy, inflamed skin patches.", "symptoms": ["Dry skin", "Intense itching", "Red inflamed patches", "Skin cracking", "Oozing or crusting"], "treatments": ["Apply moisturizers regularly to keep skin hydrated", "Use prescribed topical corticosteroids to reduce inflammation", "Avoid known triggers like harsh soaps and allergens", "Take antihistamines to relieve itching", "Consult a dermatologist for persistent or severe cases"], "severity_base": 4, "urgency": "Schedule appointment within 2-3 weeks", "urgency_level": "mid"},
    "Melanoma": {"description": "A serious and aggressive form of skin cancer. Early detection is critical for survival.", "symptoms": ["Asymmetric mole", "Irregular jagged border", "Multiple colors in lesion", "Diameter larger than 6mm", "Evolving or changing appearance"], "treatments": ["Seek immediate dermatologist evaluation — do not delay", "Surgical excision is the primary treatment", "Sentinel lymph node biopsy may be required", "Immunotherapy or targeted therapy for advanced stages", "Regular follow-up skin checks every 3-6 months"], "severity_base": 9, "urgency": "Seek medical attention immediately", "urgency_level": "high"},
    "Atopic Dermatitis": {"description": "A chronic inflammatory skin disease causing intense itching and recurring rashes.", "symptoms": ["Intense itching", "Red or brownish-gray patches", "Small raised bumps", "Thickened or scaly skin", "Raw swollen skin from scratching"], "treatments": ["Use emollients and moisturizers multiple times daily", "Apply topical corticosteroids during flare-ups", "Consider dupilumab biologic for moderate-to-severe cases", "Identify and avoid personal triggers", "Wet wrap therapy during severe flares"], "severity_base": 5, "urgency": "Schedule appointment within 1-2 weeks", "urgency_level": "mid"},
    "Basal Cell Carcinoma": {"description": "The most common form of skin cancer. Grows slowly but requires prompt treatment.", "symptoms": ["Pearly or waxy bump", "Flat flesh-colored lesion", "Bleeding or scabbing sore", "Scar-like growth", "Pink ring-shaped growth"], "treatments": ["Surgical excision to remove the tumor with margins", "Mohs surgery for high-risk or facial lesions", "Topical imiquimod for superficial type", "Radiation therapy if surgery is not possible", "Regular skin cancer screenings after treatment"], "severity_base": 7, "urgency": "Schedule appointment within 1 week", "urgency_level": "high"},
    "Melanocytic Nevi": {"description": "Commonly known as moles. Usually benign but should be monitored for changes.", "symptoms": ["Round symmetrical shape", "Uniform single color", "Smooth well-defined border", "Small diameter under 6mm", "Stable unchanging appearance"], "treatments": ["Monitor moles using the ABCDE rule regularly", "Photograph moles to track any changes over time", "Surgical removal if suspicious", "Biopsy if any concerning changes are noted", "Annual full-body skin exam by a dermatologist"], "severity_base": 2, "urgency": "Routine check at next dermatologist visit", "urgency_level": "low"},
    "Benign Keratosis": {"description": "Non-cancerous skin growths appearing as waxy or scaly patches. Very common with aging.", "symptoms": ["Waxy brown or black patches", "Rough or wart-like texture", "Stuck-on appearance", "Variable tan to dark color", "Well-defined edges"], "treatments": ["No treatment required if asymptomatic", "Cryotherapy using liquid nitrogen to remove", "Electrosurgery or curettage for removal", "Laser treatment for cosmetic removal", "Biopsy only if diagnosis is uncertain"], "severity_base": 2, "urgency": "Routine check at next appointment", "urgency_level": "low"},
    "Psoriasis": {"description": "A chronic autoimmune condition causing rapid skin cell buildup resulting in scales and red patches.", "symptoms": ["Silvery-white scales", "Dry red patches", "Cracked skin that may bleed", "Itching or burning sensation", "Thickened or pitted nails"], "treatments": ["Topical corticosteroids to reduce inflammation", "Vitamin D analogues to slow skin growth", "Phototherapy with UVB light", "Biologic medications for moderate-to-severe cases", "Moisturize regularly to reduce scaling"], "severity_base": 5, "urgency": "Schedule appointment within 2-3 weeks", "urgency_level": "mid"},
    "Seborrheic Keratoses": {"description": "Harmless non-cancerous skin growths that appear with age. Completely benign.", "symptoms": ["Waxy growths", "Brown black or pale color", "Round or oval shape", "Occasional itching", "Rough or warty surface"], "treatments": ["Treatment is optional — these are completely benign", "Cryotherapy for quick painless removal", "Electrocautery to burn off the growth", "Laser removal for cosmetic reasons", "Avoid picking or scratching to prevent irritation"], "severity_base": 1, "urgency": "No urgent action required", "urgency_level": "low"},
    "Tinea Ringworm": {"description": "A fungal infection of the skin causing ring-shaped scaly rashes. Highly contagious.", "symptoms": ["Ring-shaped red rash", "Scaly or flaky patches", "Intense itching", "Central clearing of rash", "Spreading borders"], "treatments": ["Apply antifungal cream twice daily for 2-4 weeks", "Keep the affected area clean and dry", "Avoid sharing personal items like towels", "Oral antifungal medication for widespread cases", "Continue treatment for 2 weeks after symptoms resolve"], "severity_base": 3, "urgency": "Start treatment within 1 week", "urgency_level": "mid"},
    "Warts Molluscum": {"description": "Viral skin infections causing small raised bumps. Spread by skin-to-skin contact.", "symptoms": ["Small round firm bumps", "Flesh-colored lesions", "Pearly appearance with central dimple", "Clusters of bumps", "Mild itching around bumps"], "treatments": ["Cryotherapy to freeze and destroy lesions", "Topical salicylic acid applied daily at home", "Cantharidin applied by a physician", "Laser treatment for resistant cases", "Allow natural resolution in children"], "severity_base": 3, "urgency": "Schedule appointment within 2-4 weeks", "urgency_level": "mid"},
}

@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        return tf.keras.models.load_model("skin_disease_model_best.h5")
    except:
        return None

# ── BANNER ──
st.markdown('<div class="top-banner">⚕️ &nbsp;<strong>Medical AI Tool</strong> — For informational purposes only. Always consult a licensed dermatologist.</div>', unsafe_allow_html=True)

# ── HEADER ──
st.markdown("""
<div style="background:white;border-bottom:1px solid #dde5ef;padding:14px 32px;display:flex;align-items:center;gap:14px;box-shadow:0 1px 4px rgba(15,40,80,0.07);margin-bottom:1rem;">
  <div style="width:42px;height:42px;background:linear-gradient(135deg,#1a6bcc,#0e9f8e);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px;box-shadow:0 2px 8px rgba(26,107,204,0.3);">✚</div>
  <div>
    <div style="font-family:'Playfair Display',serif;font-size:1.3rem;font-weight:700;color:#0a2d5e;">DermaCare AI</div>
    <div style="font-size:10px;color:#6b7a90;letter-spacing:0.12em;text-transform:uppercase;">Clinical Skin Analysis · Group B3</div>
  </div>
</div>""", unsafe_allow_html=True)

# ── NAV BUTTONS ──
nav1, nav2, nav_space = st.columns([1, 1, 6])
with nav1:
    if st.button("🏠 Home", use_container_width=True,
                 type="primary" if st.session_state.page == "Home" else "secondary"):
        st.session_state.page = "Home"
        st.session_state.analysis_result = None
        st.rerun()
with nav2:
    if st.button("👥 About", use_container_width=True,
                 type="primary" if st.session_state.page == "About" else "secondary"):
        st.session_state.page = "About"
        st.rerun()

st.markdown("---")

# ════════════════════════════════
# PAGE: ABOUT
# ════════════════════════════════
if st.session_state.page == "About":

    st.markdown("""
    <div class="about-hero">
      <div style="display:inline-flex;align-items:center;gap:7px;background:rgba(14,159,142,0.15);border:1px solid rgba(14,159,142,0.35);color:#4dd9c8;font-size:11px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;padding:5px 14px;border-radius:20px;margin-bottom:18px;">● Final Year Project 2026</div>
      <h1>DermaCare AI</h1>
      <p>An AI-powered clinical skin disease detection system built using deep learning and EfficientNet. Detects 10 skin conditions entirely offline — no internet, no API, no data sharing required.</p>
    </div>""", unsafe_allow_html=True)

    ca, cb, cc, cd = st.columns(4)
    for col, icon, title, val in [
        (ca, "🏫", "Department", "Computer Applications"),
        (cb, "📅", "Academic Year", "2023 – 2026"),
        (cc, "🧠", "Technology", "EfficientNet + TensorFlow"),
        (cd, "👥", "Group", "B3"),
    ]:
        with col:
            st.markdown(f'<div class="panel" style="text-align:center;padding:20px;"><div style="font-size:28px;margin-bottom:8px;">{icon}</div><div style="font-size:11px;color:#9aaabb;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">{title}</div><div style="font-weight:700;font-size:13px;color:#1a2332;">{val}</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="font-family:\'Playfair Display\',serif;font-size:1.5rem;font-weight:700;color:#1a2332;margin:24px 0 18px;text-align:center;">👥 Our Team — Group B3</div>', unsafe_allow_html=True)

    members = [
        ("Arnab Guchait",        "🧑‍💻", ["Python", "TensorFlow", "Deep Learning"]),
        ("Atanu Gayen",          "👨‍🔬", ["Python", "NumPy", "Data Processing"]),
        ("Avijit Kushwaha",      "👨‍💻", ["Streamlit", "UI Design", "Python"]),
        ("Aryan Maity",          "🧑‍🔬", ["EfficientNet", "Keras", "Model Training"]),
        ("Hrithika Chakraborty", "👩‍💻", ["Dataset Curation", "PIL", "Image Processing"]),
        ("Hiranmay Patra",       "👨‍🔬", ["Model Evaluation", "TensorFlow", "Testing"]),
        ("Ishan Sanyal",         "🧑‍💻", ["Documentation", "Python", "Deployment"]),
    ]

    cols4 = st.columns(4)
    for i, (name, avatar, techs) in enumerate(members):
        with cols4[i % 4]:
            tags = "".join([f'<span class="tech-tag">{t}</span>' for t in techs])
            st.markdown(f'<div class="member-card"><div class="member-avatar">{avatar}</div><div class="member-name">{name}</div><div class="member-role">Team Member · Group B3</div><div class="member-tech">{tags}</div></div><br/>', unsafe_allow_html=True)

    st.markdown('<div style="font-family:\'Playfair Display\',serif;font-size:1.5rem;font-weight:700;color:#1a2332;margin:16px 0 18px;text-align:center;">🛠️ Technologies Used</div>', unsafe_allow_html=True)

    techs_list = [
        ("🧠", "TensorFlow & Keras", "Deep learning framework used to build and train the EfficientNet skin disease classification model."),
        ("🖼️", "EfficientNet", "State-of-the-art CNN architecture pretrained on ImageNet, fine-tuned for skin disease detection."),
        ("🐍", "Python", "Core programming language for the entire project — from data processing to model inference."),
        ("🎨", "Streamlit", "Used to build the interactive medical-grade web interface running entirely on localhost."),
        ("🖼️", "Pillow (PIL)", "Image loading, resizing, and preprocessing for model input preparation."),
        ("📊", "NumPy", "Numerical computing for array operations and prediction post-processing."),
    ]
    tc1, tc2 = st.columns(2)
    for i, (icon, name, desc) in enumerate(techs_list):
        with (tc1 if i % 2 == 0 else tc2):
            st.markdown(f'<div class="tech-card"><div class="tech-icon">{icon}</div><div><div class="tech-name">{name}</div><div class="tech-desc">{desc}</div></div></div>', unsafe_allow_html=True)

    st.markdown('<div style="background:#0a2d5e;color:rgba(255,255,255,0.5);text-align:center;padding:22px;border-radius:12px;margin-top:24px;font-size:12.5px;"><strong style="color:rgba(255,255,255,0.8);">DermaCare AI</strong> — Group B3 · Final Year Project 2026 &nbsp;·&nbsp; Built with Python, TensorFlow &amp; Streamlit &nbsp;·&nbsp; Not a substitute for medical advice</div>', unsafe_allow_html=True)

# ════════════════════════════════
# PAGE: HOME
# ════════════════════════════════
else:
    st.markdown("""
    <div class="hero-section">
      <div class="hero-badge">● AI-Powered Dermatology</div>
      <div class="hero-title">Advanced Skin Condition<br/><span>Analysis & Detection</span></div>
      <div class="hero-desc">Upload a photo of any skin condition and receive an instant AI-powered clinical assessment across 10 disease categories — running entirely on your local machine.</div>
      <div class="hero-stats">
        <div><div class="stat-num">10</div><div class="stat-label">Conditions</div></div>
        <div><div class="stat-num">100%</div><div class="stat-label">Offline</div></div>
        <div><div class="stat-num">Free</div><div class="stat-label">No API needed</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    model = load_model()
    col_left, col_right = st.columns([1, 1.4], gap="large")

    with col_left:
        st.markdown('<div class="panel"><div class="panel-title">📤 Upload Skin Image</div><div class="panel-sub">JPG, PNG, JPEG — analyzed locally, never uploaded to any server</div></div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Drop image here or click to browse", type=["jpg","png","jpeg"], label_visibility="collapsed")

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                run_btn = st.button("🔬 Run Analysis", use_container_width=True, type="primary", disabled=st.session_state.model_running)
            with btn_col2:
                stop_btn = st.button("⏹️ Stop", use_container_width=True, type="secondary", disabled=not st.session_state.model_running)

            if stop_btn:
                st.session_state.model_running = False
                st.session_state.analysis_result = None
                st.warning("⏹️ Analysis stopped.")
                st.rerun()

            if run_btn:
                if model is None:
                    st.error("❌ Model not found. Place `skin_disease_model_best.h5` in the same folder.")
                else:
                    st.session_state.model_running = True
                    st.session_state.analysis_result = None
                    with st.spinner("🔬 Analyzing skin condition..."):
                        from tensorflow.keras.applications.efficientnet import preprocess_input
                        img = image.resize((224, 224))
                        img_array = np.array(img, dtype=np.float32)
                        img_array = preprocess_input(img_array)
                        img_array = np.expand_dims(img_array, axis=0)
                        preds = model.predict(img_array, verbose=0)[0]
                    st.session_state.model_running = False
                    st.session_state.analysis_result = preds
                    st.rerun()

        st.markdown("""
        <div style="display:flex;flex-direction:column;gap:10px;margin-top:16px;">
          <div class="panel" style="padding:14px 16px;margin-bottom:0;"><div style="display:flex;gap:10px;align-items:flex-start;"><div style="font-size:16px;">🔒</div><div><div style="font-weight:700;font-size:13px;color:#1a2332;margin-bottom:2px;">100% Private &amp; Offline</div><div style="font-size:12px;color:#6b7a90;">Your images never leave your computer.</div></div></div></div>
          <div class="panel" style="padding:14px 16px;margin-bottom:0;"><div style="display:flex;gap:10px;align-items:flex-start;"><div style="font-size:16px;">🧬</div><div><div style="font-weight:700;font-size:13px;color:#1a2332;margin-bottom:2px;">EfficientNet Deep Learning</div><div style="font-size:12px;color:#6b7a90;">Trained on thousands of dermatology images.</div></div></div></div>
          <div class="panel" style="padding:14px 16px;margin-bottom:0;"><div style="display:flex;gap:10px;align-items:flex-start;"><div style="font-size:16px;">⚕️</div><div><div style="font-weight:700;font-size:13px;color:#1a2332;margin-bottom:2px;">Clinical Disclaimer</div><div style="font-size:12px;color:#6b7a90;">Always consult a qualified dermatologist.</div></div></div></div>
        </div>""", unsafe_allow_html=True)

    with col_right:
        if uploaded_file is None:
            st.markdown('<div class="panel" style="text-align:center;padding:60px 30px;"><div style="font-size:52px;opacity:0.2;margin-bottom:16px;">🔬</div><div style="font-family:\'Playfair Display\',serif;font-size:1.2rem;color:#6b7a90;margin-bottom:8px;">No Analysis Yet</div><div style="font-size:13px;color:#9aaabb;line-height:1.7;">Upload a skin image on the left<br/>then click <strong>Run Analysis</strong>.</div></div>', unsafe_allow_html=True)

        elif st.session_state.analysis_result is None and not st.session_state.model_running:
            st.markdown('<div class="panel" style="text-align:center;padding:48px 30px;"><div style="font-size:44px;margin-bottom:14px;">⬅️</div><div style="font-family:\'Playfair Display\',serif;font-size:1.1rem;color:#6b7a90;margin-bottom:8px;">Image Ready</div><div style="font-size:13px;color:#9aaabb;line-height:1.7;">Click <strong style="color:#1a6bcc;">🔬 Run Analysis</strong> to start.<br/>Click <strong style="color:#dc2626;">⏹️ Stop</strong> to cancel anytime.</div></div>', unsafe_allow_html=True)

        elif st.session_state.analysis_result is not None:
            preds = st.session_state.analysis_result
            top3_idx = np.argsort(preds)[::-1][:3]
            primary_idx = top3_idx[0]
            primary_name = CLASS_NAMES[primary_idx]
            primary_conf = float(preds[primary_idx])
            info = DISEASE_INFO.get(primary_name, {})
            conf_pct = int(primary_conf * 100)

            if primary_conf < 0.40:
                st.markdown(f'<div class="low-conf-warn"><strong>⚠️ Low Confidence ({conf_pct}%) — Unclear Image</strong>The model could not make a confident prediction. Please upload a clearer, better-lit photo.</div>', unsafe_allow_html=True)
            else:
                sev_score = min(10, max(1, info.get("severity_base", 5)))
                severity = "Mild" if sev_score <= 3 else "Moderate" if sev_score <= 6 else "Severe"
                conf_label = "High" if conf_pct >= 70 else "Moderate" if conf_pct >= 40 else "Low"
                sev_pill = "pill-green" if severity == "Mild" else "pill-amber" if severity == "Moderate" else "pill-red"
                urgency_class = f"urgency-{info.get('urgency_level','mid')}"

                st.markdown(f"""
                <div class="dx-banner">
                  <div class="dx-label">Primary Diagnosis</div>
                  <div class="dx-name">{primary_name}</div>
                  <div class="dx-pills">
                    <div class="pill pill-blue">🔬 Confidence: {conf_pct}%</div>
                    <div class="pill {sev_pill}">⚡ {severity}</div>
                    <div class="pill pill-blue">📊 {conf_label} Confidence</div>
                  </div>
                  <div class="conf-bar-label"><span>AI Confidence Score</span><span>{conf_label}</span></div>
                  <div class="conf-bar-track"><div class="conf-bar-fill" style="width:{conf_pct}%"></div></div>
                </div>""", unsafe_allow_html=True)

                tab1, tab2, tab3, tab4 = st.tabs(["📊 Predictions", "🩺 Symptoms", "💊 Treatment", "🔍 Details"])

                with tab1:
                    st.markdown('<div class="panel">', unsafe_allow_html=True)
                    for i, idx in enumerate(top3_idx):
                        name = CLASS_NAMES[idx]; c = int(float(preds[idx]) * 100)
                        d = DISEASE_INFO.get(name, {}); medals = ["🥇","🥈","🥉"]
                        bc = ["pred-bar-1","pred-bar-2","pred-bar-3"][i]
                        st.markdown(f'<div class="pred-item"><div class="pred-row"><div class="pred-name">{medals[i]} {name}</div><div class="pred-pct">{c}%</div></div><div class="pred-bar-track"><div class="pred-bar-fill {bc}" style="width:{c}%"></div></div><div class="pred-desc">{d.get("description","")}</div></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with tab2:
                    st.markdown('<div class="panel">', unsafe_allow_html=True)
                    chips = "".join([f'<div class="chip">{s}</div>' for s in info.get("symptoms",[])])
                    st.markdown(f'<div class="chip-group">{chips}</div><div style="margin-top:16px;" class="{urgency_class}">🗓 {info.get("urgency","Consult a dermatologist")}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with tab3:
                    st.markdown('<div class="panel">', unsafe_allow_html=True)
                    for i, t in enumerate(info.get("treatments",[])):
                        st.markdown(f'<div class="treat-item"><div class="treat-num">{i+1}</div><div class="treat-text">{t}</div></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with tab4:
                    st.markdown('<div class="panel">', unsafe_allow_html=True)
                    blocks = "".join([f'<div class="sev-block {"sev-green" if i<=sev_score and sev_score<=3 else "sev-amber" if i<=sev_score and sev_score<=6 else "sev-red" if i<=sev_score else ""}"></div>' for i in range(1,11)])
                    st.markdown(f'<p style="font-size:12px;color:#6b7a90;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;">Severity Score</p><div style="font-family:\'Playfair Display\',serif;font-size:2.2rem;font-weight:700;color:#1a2332;">{sev_score} / 10</div><div class="sev-meter">{blocks}</div><div class="sev-labels"><span>Mild</span><span>Moderate</span><span>Severe</span></div><p style="font-size:12px;color:#6b7a90;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin:18px 0 12px;">About This Condition</p><p style="font-size:13.5px;color:#3d4f66;line-height:1.7;">{info.get("description","")}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="disclaimer"><strong>⚠️ Medical Disclaimer</strong>This AI analysis is for informational purposes only. Please consult a licensed dermatologist for proper diagnosis and treatment.</div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;margin-bottom:24px;">
      <div style="font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:700;color:#1a2332;margin-bottom:6px;">Clinical-Grade Local AI</div>
      <div style="font-size:14px;color:#6b7a90;">Runs entirely on your machine — no internet, no API, no data sharing</div>
    </div>
    <div class="feat-grid">
      <div class="feat-card"><div class="feat-icon feat-icon-blue">🧬</div><div class="feat-title">10 Disease Categories</div><div class="feat-desc">Detects Eczema, Melanoma, Psoriasis, Basal Cell Carcinoma, Tinea, Warts, and 4 more.</div></div>
      <div class="feat-card"><div class="feat-icon feat-icon-teal">🔒</div><div class="feat-title">Fully Offline &amp; Private</div><div class="feat-desc">No internet needed. Your images never leave your computer. Completely private.</div></div>
      <div class="feat-card"><div class="feat-icon feat-icon-amber">💊</div><div class="feat-title">Treatment Guidance</div><div class="feat-desc">Clinical recommendations and urgency ratings for each detected condition.</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="background:#0a2d5e;color:rgba(255,255,255,0.5);text-align:center;padding:22px;border-radius:12px;margin-top:32px;font-size:12.5px;"><strong style="color:rgba(255,255,255,0.8);">DermaCare AI</strong> — Group B3 · Final Year Project 2026 &nbsp;·&nbsp; Powered by EfficientNet + TensorFlow &nbsp;·&nbsp; Not a substitute for medical advice</div>', unsafe_allow_html=True)
