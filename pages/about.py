import streamlit as st
from src.style import apply_style, footer


def show():
    apply_style()

    st.markdown("""
    <link rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">

    <style>
    /* ── Hero ── */
    .hero {
        background: linear-gradient(135deg, #1a5276 0%, #17a589 100%);
        border-radius: 20px;
        padding: 48px 40px;
        text-align: center;
        color: white;
        margin-bottom: 40px;
        box-shadow: 0 8px 32px rgba(23,165,137,0.25);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -80px; left: -40px;
        width: 280px; height: 280px;
        background: rgba(255,255,255,0.04);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        font-size: 15px;
        opacity: 0.85;
        margin-bottom: 4px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 20px;
        padding: 5px 16px;
        font-size: 13px;
        margin-top: 14px;
    }

    /* ── Section titles ── */
    .about-section-title {
        font-size: 20px;
        font-weight: 700;
        color: #1a5276;
        border-left: 4px solid #17a589;
        padding-left: 12px;
        margin: 36px 0 20px 0;
    }

    /* ── KDD Pipeline ── */
    .kdd-wrapper {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0;
        flex-wrap: nowrap;
        margin-bottom: 12px;
    }
    .kdd-step {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px 16px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border-top: 4px solid #17a589;
        width: 160px;
        flex-shrink: 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
        cursor: default;
    }
    .kdd-step:hover {
        transform: translateY(-8px) scale(1.04);
        box-shadow: 0 12px 32px rgba(23,165,137,0.22);
        border-color: #1a5276;
    }
    .kdd-step i {
        font-size: 28px;
        color: #17a589;
        display: block;
        margin-bottom: 10px;
        transition: color 0.3s;
    }
    .kdd-step:hover i { color: #1a5276; }
    .kdd-num {
        font-size: 11px;
        font-weight: 700;
        color: #17a589;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .kdd-label {
        font-size: 13px;
        font-weight: 600;
        color: #1a5276;
    }
    .kdd-desc {
        font-size: 11px;
        color: #888;
        margin-top: 4px;
    }
    .kdd-arrow {
        font-size: 22px;
        color: #a8d8d0;
        margin: 0 4px;
        flex-shrink: 0;
        padding-bottom: 20px;
    }

    /* ── Page guide cards ── */
    .page-cards-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 14px;
        margin-bottom: 8px;
    }
    .page-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 22px 16px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        border-bottom: 4px solid #17a589;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    .page-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(135deg, #17a589, #1a5276);
        opacity: 0;
        transition: opacity 0.3s ease;
        border-radius: 14px;
    }
    .page-card:hover {
        transform: translateY(-10px) scale(1.04);
        box-shadow: 0 16px 40px rgba(23,165,137,0.28);
        border-bottom-color: #1a5276;
    }
    .page-card:hover::before { opacity: 1; }
    .page-card:hover .pc-icon,
    .page-card:hover .pc-name,
    .page-card:hover .pc-desc { color: white !important; position: relative; z-index: 1; }
    .pc-icon {
        font-size: 30px;
        color: #17a589;
        display: block;
        margin-bottom: 10px;
        transition: color 0.3s;
        position: relative; z-index: 1;
    }
    .pc-name {
        font-size: 13px;
        font-weight: 700;
        color: #1a5276;
        margin-bottom: 6px;
        transition: color 0.3s;
        position: relative; z-index: 1;
    }
    .pc-desc {
        font-size: 11px;
        color: #888;
        line-height: 1.4;
        transition: color 0.3s;
        position: relative; z-index: 1;
    }

    /* ── Tech stack chips ── */
    .tech-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 8px;
    }
    .tech-chip {
        border-radius: 50px;
        padding: 10px 20px;
        display: flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border: 1.5px solid #e0e7ef;
        transition: all 0.25s ease;
        cursor: default;
    }
    .tech-chip:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(23,165,137,0.18);
        border-color: #17a589;
        background: #f0faf8;
    }
    .tech-chip i { font-size: 20px; color: #17a589; }
    .tech-chip span { font-size: 13px; font-weight: 600; color: #1a5276; }

    /* ── Dataset card ── */
    .dataset-card {
        border-radius: 14px;
        padding: 24px 28px;
        border: 1.5px solid #a8d8d0;
        display: flex;
        align-items: flex-start;
        gap: 20px;
        margin-bottom: 8px;
    }
    .dataset-card i { font-size: 36px; color: #17a589; flex-shrink: 0; margin-top: 2px; }
    .dataset-card h4 { font-size: 15px; font-weight: 700; color: #1a5276; margin-bottom: 6px; }
    .dataset-card p { font-size: 13px; color: #555; line-height: 1.6; margin: 0; }

    /* ── Money rain button ── */
    .money-btn-wrap { text-align: center; margin: 40px 0 20px 0; }
    .money-btn {
        background: linear-gradient(135deg, #17a589, #1a5276);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 16px 48px;
        font-size: 18px;
        font-weight: 700;
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(23,165,137,0.35);
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
    }
    .money-btn:hover {
        transform: scale(1.06);
        box-shadow: 0 8px 30px rgba(23,165,137,0.5);
    }
    .money-btn:active { transform: scale(0.97); }

    /* ── Falling money ── */
    .money-container {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 9999;
        overflow: hidden;
    }
    .money-emoji {
        position: absolute;
        font-size: 32px;
        animation: fall linear forwards;
        top: -60px;
    }
    @keyframes fall {
        0%   { transform: translateY(0) rotate(0deg); opacity: 1; }
        80%  { opacity: 1; }
        100% { transform: translateY(110vh) rotate(720deg); opacity: 0; }
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-title">🏪 Superstore KDD Analytics Dashboard</div>
        <div class="hero-sub" style="font-size:17px; font-weight:600; margin-top:8px;">Aseel Yaghi</div>
        <div class="hero-sub">DAN614 – Advanced Data Visualization</div>
        <div class="hero-sub">MSDA · LAU Adnan Kassar School of Business</div>
        <div class="hero-badge">📊 Knowledge Discovery in Databases (KDD) Pipeline</div>
    </div>
    """, unsafe_allow_html=True)

    # ── KDD Pipeline ─────────────────────────────────────────────────────────
    st.markdown('<div class="about-section-title">The KDD Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="kdd-wrapper">
        <div class="kdd-step">
            <i class="bi bi-cloud-upload-fill"></i>
            <div class="kdd-num">Step 1</div>
            <div class="kdd-label">Data Collection</div>
            <div class="kdd-desc">Load & validate the Superstore dataset</div>
        </div>
        <div class="kdd-arrow">›</div>
        <div class="kdd-step">
            <i class="bi bi-funnel-fill"></i>
            <div class="kdd-num">Step 2</div>
            <div class="kdd-label">Data Cleaning</div>
            <div class="kdd-desc">Handle missing values, outliers & formats</div>
        </div>
        <div class="kdd-arrow">›</div>
        <div class="kdd-step">
            <i class="bi bi-bar-chart-fill"></i>
            <div class="kdd-num">Step 3</div>
            <div class="kdd-label">Descriptive Analysis</div>
            <div class="kdd-desc">Explore trends, sales & customer patterns</div>
        </div>
        <div class="kdd-arrow">›</div>
        <div class="kdd-step">
            <i class="bi bi-cpu-fill"></i>
            <div class="kdd-num">Step 4</div>
            <div class="kdd-label">Machine Learning</div>
            <div class="kdd-desc">Predict sales with regression models</div>
        </div>
        <div class="kdd-arrow">›</div>
        <div class="kdd-step">
            <i class="bi bi-lightbulb-fill"></i>
            <div class="kdd-num">Step 5</div>
            <div class="kdd-label">Insights</div>
            <div class="kdd-desc">Feature importance & business interpretation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Page Guide ────────────────────────────────────────────────────────────
    st.markdown('<div class="about-section-title">App Pages</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="page-cards-grid">
        <div class="page-card">
            <i class="bi bi-cloud-upload-fill pc-icon"></i>
            <div class="pc-name">Data Collection</div>
            <div class="pc-desc">Upload your Superstore file or connect via Google Sheets</div>
        </div>
        <div class="page-card">
            <i class="bi bi-funnel-fill pc-icon"></i>
            <div class="pc-name">Data Processing & Cleaning</div>
            <div class="pc-desc">Apply 6 cleaning measures to prepare your data</div>
        </div>
        <div class="page-card">
            <i class="bi bi-bar-chart-fill pc-icon"></i>
            <div class="pc-name">Descriptive Analysis</div>
            <div class="pc-desc">Explore 3 interactive dashboards with filters</div>
        </div>
        <div class="page-card">
            <i class="bi bi-cpu-fill pc-icon"></i>
            <div class="pc-name">Machine Learning</div>
            <div class="pc-desc">Train & compare Linear Regression and Random Forest</div>
        </div>
        <div class="page-card">
            <i class="bi bi-info-circle-fill pc-icon"></i>
            <div class="pc-name">About</div>
            <div class="pc-desc">Project overview, pipeline, tools & dataset info</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tech Stack ────────────────────────────────────────────────────────────
    st.markdown('<div class="about-section-title">Tools & Technologies</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tech-grid">
        <div class="tech-chip"><i class="bi bi-filetype-py"></i><span>Python 3</span></div>
        <div class="tech-chip"><i class="bi bi-layout-wtf"></i><span>Streamlit</span></div>
        <div class="tech-chip"><i class="bi bi-table"></i><span>Pandas</span></div>
        <div class="tech-chip"><i class="bi bi-graph-up"></i><span>Plotly</span></div>
        <div class="tech-chip"><i class="bi bi-cpu"></i><span>Scikit-learn</span></div>
        <div class="tech-chip"><i class="bi bi-calculator"></i><span>NumPy</span></div>
        <div class="tech-chip"><i class="bi bi-bootstrap"></i><span>Bootstrap Icons</span></div>
        <div class="tech-chip"><i class="bi bi-google"></i><span>Google Sheets API</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dataset Info ──────────────────────────────────────────────────────────
    st.markdown('<div class="about-section-title">Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="dataset-card">
        <i class="bi bi-database-fill"></i>
        <div>
            <h4>Superstore Sales Dataset</h4>
            <p>
                A widely-used retail analytics dataset containing <b>9,994 orders</b> across
                <b>21 columns</b> including Order ID, Customer details, Product Category,
                Sales, Profit, Discount, and Shipping information.
                It covers <b>4 years of transactions (2017–2020)</b> across the United States,
                spanning 3 product categories: Furniture, Office Supplies, and Technology.
                Ideal for KDD pipeline demonstrations, business intelligence dashboards,
                and predictive analytics projects.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # ── Money Rain Button ─────────────────────────────────────────────────────
    st.markdown('<div class="about-section-title">You Made It This Far...</div>', unsafe_allow_html=True)

    st.components.v1.html("""
    <div class="money-btn-wrap" style="text-align:center; margin: 30px 0;">
        <button class="money-btn" onclick="makeItRain()" style="
            background: linear-gradient(135deg, #17a589, #1a5276);
            color: white; border: none; border-radius: 50px;
            padding: 16px 48px; font-size: 18px; font-weight: 700;
            cursor: pointer; box-shadow: 0 4px 20px rgba(23,165,137,0.35);
            transition: all 0.3s ease; font-family: Inter, sans-serif;
            letter-spacing: 0.5px;">
            💸 Make It Rain
        </button>
        <div id="money-container" style="
            position: fixed; top: 0; left: 0;
            width: 100%; height: 100%;
            pointer-events: none; z-index: 9999; overflow: hidden;">
        </div>
        <div id="message" style="
            margin-top: 20px; font-size: 18px; font-weight: 700;
            color: #17a589; font-family: Inter, sans-serif;
            opacity: 0; transition: opacity 0.5s ease;">
        </div>
    </div>

    <style>
    .money-btn:hover {
        transform: scale(1.06);
        box-shadow: 0 8px 30px rgba(23,165,137,0.5) !important;
    }
    @keyframes fall {
        0%   { transform: translateY(0) rotate(0deg) scale(1);   opacity: 1; }
        50%  { transform: translateY(50vh) rotate(360deg) scale(1.2); opacity: 1; }
        100% { transform: translateY(110vh) rotate(720deg) scale(0.8); opacity: 0; }
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.8); }
        to   { opacity: 1; transform: scale(1); }
    }
    </style>

    <script>
    const emojis = ['💸','💰','🤑','💵','💴','💶','💷','🪙','✨','⭐','🎉','🏆'];
    let raining = false;

    function makeItRain() {
        if (raining) return;
        raining = true;

        const container = document.getElementById('money-container');
        const messages  = [
            "💯 A+ is loading...",
            "🎓 Graduation incoming!",
            "🏆 Best project ever!",
            "🚀 Straight to the top!",
            "✨ Professor approved!"
        ];
        const msg = document.getElementById('message');
        msg.innerText = messages[Math.floor(Math.random() * messages.length)];
        msg.style.opacity = '1';

        const count = 80;
        for (let i = 0; i < count; i++) {
            setTimeout(() => {
                const el = document.createElement('div');
                el.style.cssText = `
                    position: fixed;
                    left: ${Math.random() * 100}vw;
                    top: -60px;
                    font-size: ${20 + Math.random() * 28}px;
                    pointer-events: none;
                    z-index: 9999;
                    animation: fall ${1.5 + Math.random() * 2}s linear forwards;
                `;
                el.innerText = emojis[Math.floor(Math.random() * emojis.length)];
                document.body.appendChild(el);
                setTimeout(() => el.remove(), 3500);
            }, i * 40);
        }

        setTimeout(() => {
            raining = false;
            msg.style.opacity = '0';
        }, 5000);
    }
    </script>
    """, height=140)

    footer()