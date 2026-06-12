import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.title("Resume Screening System")

job_description = st.text_area("Enter Job Description")
resume_text = st.text_area("Paste Resume Content")

if st.button("Match Resume"):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    jd_embedding = model.encode(job_description, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)

    similarity = util.cos_sim(jd_embedding, resume_embedding)

    score = float(similarity[0][0]) * 100

    st.success(f"Match Score: {score:.2f}%")
