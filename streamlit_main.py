import streamlit as st  
import requests  
  
st.title('Emotion classifier and ChatDev paper Q&A')  
  
option = st.selectbox('Choose an option:', ['Emotion Classification', 'ChatDev paper Q&A'])  
  
if option == 'Emotion Classification':  
    st.header('Emotion Classification')  
    emotion_input = st.text_input('Enter a text to classify its emotion:')  
    if st.button('Classify Emotion'):  
        response = requests.get(f'http://127.0.0.1:8000/emotion?user_input={emotion_input}')  
        st.write('Emotion:', response.json())  
  
elif option == 'ChatDev paper Q&A':  
    st.header('ChatDev paper Q&A')  
    rag_input = st.text_input('Enter a query about the paper:')  
    if st.button('Submit RAG Query'):  
        response = requests.get(f'http://127.0.0.1:8000/rag?user_query={rag_input}', stream=True)  
        for line in response.iter_lines():  
            if line:  
                decoded_line = line.decode('utf-8')  
                event, data = decoded_line.split(':', 1)  
                if event.strip() == 'event':  
                    event_name = data.strip()  
                elif event.strip() == 'data' and event_name == 'generatingQuery':  
                    st.write('Status:', data.strip())  
                elif event.strip() == 'data' and event_name == 'queryGenerated':  
                    st.write('Response:', data.strip())  
