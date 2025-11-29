import streamlit as st
import rag_engine  # our existing RAG engine

MODEL_NAME = "llama3.2:1b"  # you can change later if you use another model

st.set_page_config(page_title="LLM RAG Demo", page_icon="🧠")

st.title("🧠 LLM RAG Evaluation Demo")
st.write("Ask questions based on your PDF documents. Choose with/without RAG to compare.")

# --- Sidebar ---
st.sidebar.header("Settings")
mode = st.sidebar.radio("Answer mode:", ["RAG (with documents)", "Baseline (no RAG)"])
top_k = st.sidebar.slider("Top-k chunks (for RAG)", 1, 10, 5)

st.sidebar.markdown("---")
st.sidebar.write("Model:", MODEL_NAME)

# --- Main input ---
question = st.text_area("💬 Enter your question:", height=100)

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):
            if mode.startswith("RAG"):
                # Use our RAG pipeline
                answer = rag_engine.rag_answer(question, model=MODEL_NAME)
                st.subheader("📚 RAG Answer")
                st.write(answer)
            else:
                # Baseline: no RAG, just the LLM
                prompt = f"""You are a helpful assistant.
Answer the following question using your internal knowledge only.

Question: {question}

Answer:"""
                answer = rag_engine.call_ollama(MODEL_NAME, prompt)
                st.subheader("🤖 Baseline LLM Answer (no RAG)")
                st.write(answer)

        st.markdown("---")
        st.caption("Tip: Try the same question in both modes and compare.")
