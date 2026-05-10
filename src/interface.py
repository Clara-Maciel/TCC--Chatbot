import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from chat_engine import responder

st.set_page_config(page_title="IA IF Baiano", page_icon="🎓", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎓 Assistente Acadêmico IF Baiano")
st.markdown("Bem-vindo! Estou aqui para auxiliar com regulamentos, normas acadêmicas e procedimentos institucionais.")

col1, col2 = st.columns([2.5, 1])

with col2:
    st.subheader("📖 Guia Acadêmico")
    st.info("Posso tirar dúvidas sobre:")
    st.write("- **Estágios:** NRI e Procedimentos")
    st.write("- **Regras:** Deveres e direitos dos estudantes")
    st.write("- **SUAP:** Como abrir processos acadêmicos")
    if st.button("🗑️ Limpar conversa"):
        st.session_state.messages = []
        st.rerun()

with col1:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Como posso ajudar?")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Passa o histórico SEM a pergunta atual (ela já vai no argumento `pergunta`)
        historico_anterior = st.session_state.messages[:-1]

        with st.chat_message("assistant"):
            with st.spinner("Consultando documentação..."):
                resposta = responder(prompt, historico_anterior)
                st.markdown(resposta)

        st.session_state.messages.append({"role": "assistant", "content": resposta})
        st.rerun()
