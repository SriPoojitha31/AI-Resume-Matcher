import streamlit as st
from resume_parser import extract_text_from_file
from jd_matcher import compute_similarity, compute_bert_similarity
from utils.text_cleaning import clean_text
from utils.file_utils import extract_entities_from_text, extract_skills_from_jd, generate_pdf_report
import pandas as pd
import io
import spacy
import subprocess
import sys

def extract_entities_from_text(text):
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        try:
            subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm', '--user'], check=True)
            nlp = spacy.load('en_core_web_sm')
        except Exception as e:
            return {"skills": [], "experience": [], "location": []}
    try:
        doc = nlp(text)
        skills = set()
        experience = set()
        location = set()
        for ent in doc.ents:
            if ent.label_ in ["SKILL", "SKILLS"]:
                skills.add(ent.text)
            elif ent.label_ in ["EXPERIENCE", "DATE"]:
                experience.add(ent.text)
            elif ent.label_ in ["GPE", "LOC", "LOCATION"]:
                location.add(ent.text)
        return {"skills": list(skills), "experience": list(experience), "location": list(location)}
    except Exception:
        return {"skills": [], "experience": [], "location": []}
    # FINAL fallback, should never be hit, but guarantees a return value
    return {"skills": [], "experience": [], "location": []}

# --- Streamlit Theme ---
st.set_page_config(
    page_title="AI Resume Matcher",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    body, .stApp { background-color: #101c14 !important; }
    .stApp, .css-1d391kg, .css-1v0mbdj, .css-1cpxqw2, .stTextInput, .stTextArea, .stSelectbox, .stFileUploader, .stRadio, .stButton>button {
        color: #fff !important;
        background-color: #101c14 !important;
    }
    .stSidebar, .css-1lcbmhc, .css-1d391kg, .css-1v0mbdj, .css-1cpxqw2 {
        background-color: #000 !important;
    }
    .st-bw, .st-cq, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz, .st-da, .st-db, .st-dc, .st-dd, .st-de, .st-df, .st-dg, .st-dh, .st-di, .st-dj, .st-dk, .st-dl, .st-dm, .st-dn, .st-do, .st-dp, .st-dq, .st-dr, .st-ds, .st-dt, .st-du, .st-dv, .st-dw, .st-dx, .st-dy, .st-dz, .st-e0, .st-e1, .st-e2, .st-e3, .st-e4, .st-e5, .st-e6, .st-e7, .st-e8, .st-e9, .st-ea, .st-eb, .st-ec, .st-ed, .st-ee, .st-ef, .st-eg, .st-eh, .st-ei, .st-ej, .st-ek, .st-el, .st-em, .st-en, .st-eo, .st-ep, .st-eq, .st-er, .st-es, .st-et, .st-eu, .st-ev, .st-ew, .st-ex, .st-ey, .st-ez, .st-f0, .st-f1, .st-f2, .st-f3, .st-f4, .st-f5, .st-f6, .st-f7, .st-f8, .st-f9, .st-fa, .st-fb, .st-fc, .st-fd, .st-fe, .st-ff, .st-fg, .st-fh, .st-fi, .st-fj, .st-fk, .st-fl, .st-fm, .st-fn, .st-fo, .st-fp, .st-fq, .st-fr, .st-fs, .st-ft, .st-fu, .st-fv, .st-fw, .st-fx, .st-fy, .st-fz, .st-g0, .st-g1, .st-g2, .st-g3, .st-g4, .st-g5, .st-g6, .st-g7, .st-g8, .st-g9, .st-ga, .st-gb, .st-gc, .st-gd, .st-ge, .st-gf, .st-gg, .st-gh, .st-gi, .st-gj, .st-gk, .st-gl, .st-gm, .st-gn, .st-go, .st-gp, .st-gq, .st-gr, .st-gs, .st-gt, .st-gu, .st-gv, .st-gw, .st-gx, .st-gy, .st-gz, .st-h0, .st-h1, .st-h2, .st-h3, .st-h4, .st-h5, .st-h6, .st-h7, .st-h8, .st-h9, .st-ha, .st-hb, .st-hc, .st-hd, .st-he, .st-hf, .st-hg, .st-hh, .st-hi, .st-hj, .st-hk, .st-hl, .st-hm, .st-hn, .st-ho, .st-hp, .st-hq, .st-hr, .st-hs, .st-ht, .st-hu, .st-hv, .st-hw, .st-hx, .st-hy, .st-hz, .st-i0, .st-i1, .st-i2, .st-i3, .st-i4, .st-i5, .st-i6, .st-i7, .st-i8, .st-i9, .st-ia, .st-ib, .st-ic, .st-id, .st-ie, .st-if, .st-ig, .st-ih, .st-ii, .st-ij, .st-ik, .st-il, .st-im, .st-in, .st-io, .st-ip, .st-iq, .st-ir, .st-is, .st-it, .st-iu, .st-iv, .st-iw, .st-ix, .st-iy, .st-iz, .st-j0, .st-j1, .st-j2, .st-j3, .st-j4, .st-j5, .st-j6, .st-j7, .st-j8, .st-j9, .st-ja, .st-jb, .st-jc, .st-jd, .st-je, .st-jf, .st-jg, .st-jh, .st-ji, .st-jj, .st-jk, .st-jl, .st-jm, .st-jn, .st-jo, .st-jp, .st-jq, .st-jr, .st-js, .st-jt, .st-ju, .st-jv, .st-jw, .st-jx, .st-jy, .st-jz, .st-k0, .st-k1, .st-k2, .st-k3, .st-k4, .st-k5, .st-k6, .st-k7, .st-k8, .st-k9, .st-ka, .st-kb, .st-kc, .st-kd, .st-ke, .st-kf, .st-kg, .st-kh, .st-ki, .st-kj, .st-kk, .st-kl, .st-km, .st-kn, .st-ko, .st-kp, .st-kq, .st-kr, .st-ks, .st-kt, .st-ku, .st-kv, .st-kw, .st-kx, .st-ky, .st-kz, .st-l0, .st-l1, .st-l2, .st-l3, .st-l4, .st-l5, .st-l6, .st-l7, .st-l8, .st-l9, .st-la, .st-lb, .st-lc, .st-ld, .st-le, .st-lf, .st-lg, .st-lh, .st-li, .st-lj, .st-lk, .st-ll, .st-lm, .st-ln, .st-lo, .st-lp, .st-lq, .st-lr, .st-ls, .st-lt, .st-lu, .st-lv, .st-lw, .st-lx, .st-ly, .st-lz, .st-m0, .st-m1, .st-m2, .st-m3, .st-m4, .st-m5, .st-m6, .st-m7, .st-m8, .st-m9, .st-ma, .st-mb, .st-mc, .st-md, .st-me, .st-mf, .st-mg, .st-mh, .st-mi, .st-mj, .st-mk, .st-ml, .st-mm, .st-mn, .st-mo, .st-mp, .st-mq, .st-mr, .st-ms, .st-mt, .st-mu, .st-mv, .st-mw, .st-mx, .st-my, .st-mz, .st-n0, .st-n1, .st-n2, .st-n3, .st-n4, .st-n5, .st-n6, .st-n7, .st-n8, .st-n9, .st-na, .st-nb, .st-nc, .st-nd, .st-ne, .st-nf, .st-ng, .st-nh, .st-ni, .st-nj, .st-nk, .st-nl, .st-nm, .st-nn, .st-no, .st-np, .st-nq, .st-nr, .st-ns, .st-nt, .st-nu, .st-nv, .st-nw, .st-nx, .st-ny, .st-nz, .st-o0, .st-o1, .st-o2, .st-o3, .st-o4, .st-o5, .st-o6, .st-o7, .st-o8, .st-o9, .st-oa, .st-ob, .st-oc, .st-od, .st-oe, .st-of, .st-og, .st-oh, .st-oi, .st-oj, .st-ok, .st-ol, .st-om, .st-on, .st-oo, .st-op, .st-oq, .st-or, .st-os, .st-ot, .st-ou, .st-ov, .st-ow, .st-ox, .st-oy, .st-oz, .st-p0, .st-p1, .st-p2, .st-p3, .st-p4, .st-p5, .st-p6, .st-p7, .st-p8, .st-p9, .st-pa, .st-pb, .st-pc, .st-pd, .st-pe, .st-pf, .st-pg, .st-ph, .st-pi, .st-pj, .st-pk, .st-pl, .st-pm, .st-pn, .st-po, .st-pp, .st-pq, .st-pr, .st-ps, .st-pt, .st-pu, .st-pv, .st-pw, .st-px, .st-py, .st-pz, .st-q0, .st-q1, .st-q2, .st-q3, .st-q4, .st-q5, .st-q6, .st-q7, .st-q8, .st-q9, .st-qa, .st-qb, .st-qc, .st-qd, .st-qe, .st-qf, .st-qg, .st-qh, .st-qi, .st-qj, .st-qk, .st-ql, .st-qm, .st-qn, .st-qo, .st-qp, .st-qq, .st-qr, .st-qs, .st-qt, .st-qu, .st-qv, .st-qw, .st-qx, .st-qy, .st-qz, .st-r0, .st-r1, .st-r2, .st-r3, .st-r4, .st-r5, .st-r6, .st-r7, .st-r8, .st-r9, .st-ra, .st-rb, .st-rc, .st-rd, .st-re, .st-rf, .st-rg, .st-rh, .st-ri, .st-rj, .st-rk, .st-rl, .st-rm, .st-rn, .st-ro, .st-rp, .st-rq, .st-rr, .st-rs, .st-rt, .st-ru, .st-rv, .st-rw, .st-rx, .st-ry, .st-rz, .st-s0, .st-s1, .st-s2, .st-s3, .st-s4, .st-s5, .st-s6, .st-s7, .st-s8, .st-s9, .st-sa, .st-sb, .st-sc, .st-sd, .st-se, .st-sf, .st-sg, .st-sh, .st-si, .st-sj, .st-sk, .st-sl, .st-sm, .st-sn, .st-so, .st-sp, .st-sq, .st-sr, .st-ss, .st-st, .st-su, .st-sv, .st-sw, .st-sx, .st-sy, .st-sz, .st-t0, .st-t1, .st-t2, .st-t3, .st-t4, .st-t5, .st-t6, .st-t7, .st-t8, .st-t9, .st-ta, .st-tb, .st-tc, .st-td, .st-te, .st-tf, .st-tg, .st-th, .st-ti, .st-tj, .st-tk, .st-tl, .st-tm, .st-tn, .st-to, .st-tp, .st-tq, .st-tr, .st-ts, .st-tt, .st-tu, .st-tv, .st-tw, .st-tx, .st-ty, .st-tz, .st-u0, .st-u1, .st-u2, .st-u3, .st-u4, .st-u5, .st-u6, .st-u7, .st-u8, .st-u9, .st-ua, .st-ub, .st-uc, .st-ud, .st-ue, .st-uf, .st-ug, .st-uh, .st-ui, .st-uj, .st-uk, .st-ul, .st-um, .st-un, .st-uo, .st-up, .st-uq, .st-ur, .st-us, .st-ut, .st-uu, .st-uv, .st-uw, .st-ux, .st-uy, .st-uz, .st-v0, .st-v1, .st-v2, .st-v3, .st-v4, .st-v5, .st-v6, .st-v7, .st-v8, .st-v9, .st-va, .st-vb, .st-vc, .st-vd, .st-ve, .st-vf, .st-vg, .st-vh, .st-vi, .st-vj, .st-vk, .st-vl, .st-vm, .st-vn, .st-vo, .st-vp, .st-vq, .st-vr, .st-vs, .st-vt, .st-vu, .st-vv, .st-vw, .st-vx, .st-vy, .st-vz, .st-w0, .st-w1, .st-w2, .st-w3, .st-w4, .st-w5, .st-w6, .st-w7, .st-w8, .st-w9, .st-wa, .st-wb, .st-wc, .st-wd, .st-we, .st-wf, .st-wg, .st-wh, .st-wi, .st-wj, .st-wk, .st-wl, .st-wm, .st-wn, .st-wo, .st-wp, .st-wq, .st-wr, .st-ws, .st-wt, .st-wu, .st-wv, .st-ww, .st-wx, .st-wy, .st-wz, .st-x0, .st-x1, .st-x2, .st-x3, .st-x4, .st-x5, .st-x6, .st-x7, .st-x8, .st-x9, .st-xa, .st-xb, .st-xc, .st-xd, .st-xe, .st-xf, .st-xg, .st-xh, .st-xi, .st-xj, .st-xk, .st-xl, .st-xm, .st-xn, .st-xo, .st-xp, .st-xq, .st-xr, .st-xs, .st-xt, .st-xu, .st-xv, .st-xw, .st-xx, .st-xy, .st-xz, .st-y0, .st-y1, .st-y2, .st-y3, .st-y4, .st-y5, .st-y6, .st-y7, .st-y8, .st-y9, .st-ya, .st-yb, .st-yc, .st-yd, .st-ye, .st-yf, .st-yg, .st-yh, .st-yi, .st-yj, .st-yk, .st-yl, .st-ym, .st-yn, .st-yo, .st-yp, .st-yq, .st-yr, .st-ys, .st-yt, .st-yu, .st-yv, .st-yw, .st-yx, .st-yy, .st-yz, .st-z0, .st-z1, .st-z2, .st-z3, .st-z4, .st-z5, .st-z6, .st-z7, .st-z8, .st-z9, .st-za, .st-zb, .st-zc, .st-zd, .st-ze, .st-zf, .st-zg, .st-zh, .st-zi, .st-zj, .st-zk, .st-zl, .st-zm, .st-zn, .st-zo, .st-zp, .st-zq, .st-zr, .st-zs, .st-zt, .st-zu, .st-zv, .st-zw, .st-zx, .st-zy, .st-zz { color: #fff !important; }
    .stHighlight { background-color: #183c2b !important; color: #fff !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar Navigation ---
section = st.sidebar.radio(
    "Go to section:",
    ["Job Seeker (Single Match)", "Job Seeker (Batch Match)", "Employer (Batch & Analytics)"]
)

# --- JD Templates ---
jd_templates = {
    "Python Developer": "We are looking for a Python Developer with experience in Python, SQL, data analysis, and machine learning.",
    "Data Analyst": "Seeking a Data Analyst skilled in Excel, SQL, data analysis, and communication.",
    "ML Engineer": "Hiring an ML Engineer with expertise in Python, deep learning, machine learning, and project management.",
    "Project Manager": "Project Manager needed with leadership, project management, and communication skills."
}
st.sidebar.header("JD Templates")
template_choice = st.sidebar.selectbox("Choose a JD Template", ["None"] + list(jd_templates.keys()))
if template_choice != "None":
    st.session_state["jd_textarea"] = jd_templates[template_choice]

if section == "Job Seeker (Single Match)":
    st.header("Job Seeker: Check Your Resume Against a Job Description")
    resume_file = st.file_uploader("Upload your resume (PDF, DOC, DOCX)", type=["pdf", "doc", "docx"], key="js_resume")
    jd_input = st.text_area("Paste Job Description Here", key="jd_textarea")
    jd_file = st.file_uploader("Or upload JD as .txt file", type="txt", key="js_jd_file")
    if jd_file is not None:
        jd_text = jd_file.read().decode("utf-8")
    else:
        jd_text = jd_input.strip()
    match_method = st.radio(
        "Select similarity method:",
        ["TF-IDF (Fast, Basic)", "BERT (Deep Semantic, Slower)"],
        key="js_match_method"
    )
    if resume_file and jd_text:
        resume_text = extract_text_from_file(resume_file)
        resume_text = clean_text(resume_text)
        jd_text_clean = clean_text(jd_text)
        if match_method == "BERT (Deep Semantic, Slower)":
            match_score = compute_bert_similarity(resume_text, jd_text_clean)
        else:
            match_score = compute_similarity(resume_text, jd_text_clean)
        st.metric(label="Match Score", value=f"{match_score}%", delta=None)
        st.progress(int(match_score))
        # NER and skills
        entities = extract_entities_from_text(resume_text)
        if not isinstance(entities, dict) or "skills" not in entities:
            entities = {"skills": [], "experience": [], "location": []}
        resume_skills = set([s.lower() for s in entities["skills"]])
        jd_skills = extract_skills_from_jd(jd_text_clean)
        matched_skills = sorted(resume_skills & jd_skills)
        missing_skills = sorted(jd_skills - resume_skills)
        st.markdown(f"**Matched Skills:** <span style='color:green'>{', '.join(matched_skills) if matched_skills else 'None'}</span>", unsafe_allow_html=True)
        st.markdown(f"**Top 3 Missing Skills:** <span style='color:red'>{', '.join(missing_skills[:3]) if missing_skills else 'None'}</span>", unsafe_allow_html=True)
        st.markdown(f"**All Extracted Skills:** {', '.join(entities['skills']) if entities['skills'] else 'None'}")
        st.markdown(f"**Experience:** {', '.join(entities['experience']) if entities['experience'] else 'None'}")
        st.markdown(f"**Location:** {', '.join(entities['location']) if entities['location'] else 'None'}")
        # --- Suggestions ---
        st.header("Resume Improvement Suggestions")
        if missing_skills:
            st.info(f"Consider adding these skills to your resume if you have them: {', '.join(missing_skills[:3])}")
        else:
            st.success("Your resume covers all the key skills in the JD!")
        if match_score < 50:
            st.warning("Your match score is low. Try to tailor your resume more closely to the job description.")
        elif match_score < 80:
            st.info("Your match is moderate. Consider emphasizing relevant skills and experience.")
        else:
            st.success("Great match! Your resume is highly relevant to this job.")

elif section == "Job Seeker (Batch Match)":
    st.header("Job Seeker: Batch Resume Match (Multiple JDs)")
    resume_file = st.file_uploader("Upload your resume (PDF, DOC, DOCX)", type=["pdf", "doc", "docx"], key="js_batch_resume")
    jd_files = st.file_uploader("Upload multiple JD .txt files", type="txt", accept_multiple_files=True, key="js_batch_jds")
    match_method = st.radio(
        "Select similarity method:",
        ["TF-IDF (Fast, Basic)", "BERT (Deep Semantic, Slower)"],
        key="js_batch_match_method"
    )
    if resume_file and jd_files:
        resume_text = extract_text_from_file(resume_file)
        resume_text = clean_text(resume_text)
        entities = extract_entities_from_text(resume_text)
        if not isinstance(entities, dict) or "skills" not in entities:
            entities = {"skills": [], "experience": [], "location": []}
        resume_skills = set([s.lower() for s in entities["skills"]])
        results = []
        for jd_file in jd_files:
            jd_text = jd_file.read().decode("utf-8")
            jd_text_clean = clean_text(jd_text)
            jd_skills = extract_skills_from_jd(jd_text_clean)
            matched_skills = sorted(resume_skills & jd_skills)
            missing_skills = sorted(jd_skills - resume_skills)
            if match_method == "BERT (Deep Semantic, Slower)":
                match_score = compute_bert_similarity(resume_text, jd_text_clean)
            else:
                match_score = compute_similarity(resume_text, jd_text_clean)
            results.append({
                "JD File": jd_file.name,
                "Match Score": match_score,
                "Matched Skills": ", ".join(matched_skills),
                "Top 3 Missing Skills": ", ".join(missing_skills[:3])
            })
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

elif section == "Employer (Batch & Analytics)":
    st.header("Employer: Batch Resume Evaluation & Analytics")
    tab1, tab2 = st.tabs(["Batch Resume Evaluation", "Job Role Auto-Matching"])

    with tab1:
        st.header("1. Upload Resume(s) (PDF/DOC/DOCX)")
        uploaded_resumes = st.file_uploader(
            "Choose one or more resume files (PDF, DOC, DOCX)",
            type=["pdf", "doc", "docx"],
            accept_multiple_files=True,
            key="resumes"
        )
        st.header("2. Provide Job Description")
        jd_input = st.text_area("Paste Job Description Here", height=150, key="jd_textarea")
        jd_file = st.file_uploader("Or upload JD as .txt file", type="txt", key="jd_file")
        jd_text = jd_input.strip()
        if jd_file is not None:
            jd_text = jd_file.read().decode("utf-8")
        # --- Matching Method ---
        st.header("3. Choose Matching Method")
        match_method = st.radio(
            "Select similarity method:",
            ["TF-IDF (Fast, Basic)", "BERT (Deep Semantic, Slower)"],
            key="match_method1"
        )
        if uploaded_resumes and jd_text:
            results = []
            jd_text_clean = clean_text(jd_text)
            jd_skills = extract_skills_from_jd(jd_text_clean)
            for resume_file in uploaded_resumes:
                resume_text = extract_text_from_file(resume_file)
                resume_text = clean_text(resume_text)
                if match_method == "BERT (Deep Semantic, Slower)":
                    match_score = compute_bert_similarity(resume_text, jd_text_clean)
                    method_used = "BERT"
                else:
                    match_score = compute_similarity(resume_text, jd_text_clean)
                    method_used = "TF-IDF"
                if match_score > 80:
                    status = "Great Match"
                elif match_score > 50:
                    status = "Moderate Match"
                else:
                    status = "Low Match"
                # --- NER Extraction ---
                entities = extract_entities_from_text(resume_text)
                if not isinstance(entities, dict) or "skills" not in entities:
                    entities = {"skills": [], "experience": [], "location": []}
                resume_skills = set([s.lower() for s in entities["skills"]])
                matched_skills = sorted(resume_skills & jd_skills)
                missing_skills = sorted(jd_skills - resume_skills)
                # --- Visual Match Score ---
                st.subheader(f"Match Score for {resume_file.name}")
                st.metric(label="Match Score", value=f"{match_score}%", delta=None)
                st.progress(int(match_score))
                # --- Keyword Highlighting ---
                st.markdown(
                    f"**Matched Skills:** <span style='color:green'>{', '.join(matched_skills) if matched_skills else 'None'}</span>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"**Top 3 Missing Skills:** <span style='color:red'>{', '.join(missing_skills[:3]) if missing_skills else 'None'}</span>",
                    unsafe_allow_html=True
                )
                results.append({
                    "Resume File": resume_file.name,
                    "Match Score": match_score,
                    "Status": status,
                    "Method": method_used,
                    "Skills": ", ".join(entities["skills"]),
                    "Experience": ", ".join(entities["experience"]),
                    "Location": ", ".join(entities["location"]),
                    "Matched Skills": ", ".join(matched_skills),
                    "Missing Skills": ", ".join(missing_skills)
                })
            results = sorted(results, key=lambda x: x["Match Score"], reverse=True)
            df = pd.DataFrame(results)
            # --- Analytics Dashboard ---
            st.header("4. Analytics Dashboard")
            st.metric("Number of Resumes", len(df))
            if not df.empty:
                st.metric("Average Match Score", f"{df['Match Score'].mean():.2f}%")
                # Top matched skills
                from collections import Counter
                all_matched = []
                for skills in df["Matched Skills"]:
                    all_matched.extend([s.strip() for s in skills.split(",") if s.strip()])
                top_skills = Counter(all_matched).most_common(10)
                if top_skills:
                    st.bar_chart(pd.DataFrame(top_skills, columns=["Skill", "Count"]).set_index("Skill"))
            # --- Match Results Table ---
            st.header("5. Match Results (All Resumes)")
            st.dataframe(df, use_container_width=True)
            st.header("5. Export Results")
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Results')
            st.download_button(
                label="Download Results as Excel",
                data=output.getvalue(),
                file_name="match_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            # CSV Export
            st.download_button(
                label="Download Results as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="match_results.csv",
                mime="text/csv"
            )
            # JSON Export
            st.download_button(
                label="Download Results as JSON",
                data=df.to_json(orient='records', force_ascii=False, indent=2),
                file_name="match_results.json",
                mime="application/json"
            )
            # --- PDF Report ---
            pdf_bytes = generate_pdf_report(df, title="Batch Resume Match Report")
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="match_report.pdf",
                mime="application/pdf"
            )
        else:
            st.info("Please upload at least one resume and provide a job description to see your match scores.")

    with tab2:
        st.header("1. Upload a Resume (PDF/DOC/DOCX)")
        single_resume = st.file_uploader(
            "Choose a resume file (PDF, DOC, DOCX)",
            type=["pdf", "doc", "docx"],
            key="single_resume"
        )
        st.header("2. Upload Job Descriptions (.txt, multiple)")
        jd_files = st.file_uploader("Upload one or more JD .txt files", type="txt", accept_multiple_files=True, key="multi_jd")
        st.header("3. Choose Matching Method")
        match_method2 = st.radio(
            "Select similarity method:",
            ["TF-IDF (Fast, Basic)", "BERT (Deep Semantic, Slower)"],
            key="match_method2"
        )
        if single_resume and jd_files:
            resume_text = extract_text_from_file(single_resume)
            resume_text = clean_text(resume_text)
            # --- NER Extraction for single resume ---
            entities = extract_entities_from_text(resume_text)
            if not isinstance(entities, dict) or "skills" not in entities:
                entities = {"skills": [], "experience": [], "location": []}
            resume_skills = set([s.lower() for s in entities["skills"]])
            st.markdown("**Extracted Skills:** " + ", ".join(entities["skills"]))
            st.markdown("**Extracted Experience:** " + ", ".join(entities["experience"]))
            st.markdown("**Extracted Location:** " + ", ".join(entities["location"]))
            results = []
            for jd_file in jd_files:
                jd_text = jd_file.read().decode("utf-8")
                jd_text_clean = clean_text(jd_text)
                jd_skills = extract_skills_from_jd(jd_text_clean)
                matched_skills = sorted(resume_skills & jd_skills)
                missing_skills = sorted(jd_skills - resume_skills)
                if match_method2 == "BERT (Deep Semantic, Slower)":
                    match_score = compute_bert_similarity(resume_text, jd_text_clean)
                    method_used = "BERT"
                else:
                    match_score = compute_similarity(resume_text, jd_text_clean)
                    method_used = "TF-IDF"
                if match_score > 80:
                    status = "Great Match"
                elif match_score > 50:
                    status = "Moderate Match"
                else:
                    status = "Low Match"
                # --- Visual Match Score ---
                st.subheader(f"Match Score for {jd_file.name}")
                st.metric(label="Match Score", value=f"{match_score}%", delta=None)
                st.progress(int(match_score))
                # --- Keyword Highlighting ---
                st.markdown(
                    f"**Matched Skills:** <span style='color:green'>{', '.join(matched_skills) if matched_skills else 'None'}</span>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"**Top 3 Missing Skills:** <span style='color:red'>{', '.join(missing_skills[:3]) if missing_skills else 'None'}</span>",
                    unsafe_allow_html=True
                )
                results.append({
                    "JD File": jd_file.name,
                    "Match Score": match_score,
                    "Status": status,
                    "Method": method_used,
                    "Matched Skills": ", ".join(matched_skills),
                    "Missing Skills": ", ".join(missing_skills)
                })
            results = sorted(results, key=lambda x: x["Match Score"], reverse=True)
            df = pd.DataFrame(results)
            st.header("4. Match Results (All JDs)")
            st.dataframe(df, use_container_width=True)
            st.header("5. Export Results")
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Results')
            st.download_button(
                label="Download Results as Excel",
                data=output.getvalue(),
                file_name="jd_match_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            # CSV Export
            st.download_button(
                label="Download Results as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="jd_match_results.csv",
                mime="text/csv"
            )
            # JSON Export
            st.download_button(
                label="Download Results as JSON",
                data=df.to_json(orient='records', force_ascii=False, indent=2),
                file_name="jd_match_results.json",
                mime="application/json"
            )
            # --- PDF Report ---
            pdf_bytes = generate_pdf_report(df, title="Job Role Auto-Match Report")
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="jd_match_report.pdf",
                mime="application/pdf"
            )
        else:
            st.info("Please upload a resume and at least one JD .txt file to see your match scores.")
