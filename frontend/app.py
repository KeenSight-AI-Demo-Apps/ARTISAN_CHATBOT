import streamlit as st
import requests

st.set_page_config(page_title="Artisan Support Chatbot", page_icon="🧵")
st.title("🧵 Artisan Customer Support Chatbot")
st.markdown("Ask a question like **'Do you ship to Australia?'**")

question = st.text_input("Your question")

if st.button("Ask"):
    if question.strip():
        try:
            response = requests.post(
                "http://localhost:8000/ask", json={"question": question}
            )
            st.write("Status code:", response.status_code)
            st.write("Response text:", response.text)

            if response.status_code == 200:
                data = response.json()
                st.write("🤖 Bot:", data.get("answer", "No answer in response"))
            else:
                st.write(f"Error: Backend returned status code {response.status_code}")

        except requests.exceptions.RequestException as e:
            st.write(f"Request failed: {e}")
    else:
        st.write("Please enter a question before asking.")
