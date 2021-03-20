import streamlit as st
import spacy
import spacy_streamlit

nlp = spacy.load('en_core_web_sm')

def main():
    st.title('Simple NLP app for tokenizing and Named Entity Recognition')
    menu = ['Home', 'NER']
    choice = st.sidebar.selectbox('Menu', menu)
    if choice == 'Home':
        st.subheader('Tokentization')
        raw_text = st.text_area("Your text", 'Enter your text here')
        docs = nlp(raw_text)
        if st.button('Tokenize'):
            spacy_streamlit.visualize_tokens(docs, attrs= ['text','pos_','tag_','ent_type_'])


    elif choice == 'NER':
        st.subheader('Named Entity Recognition')
        raw_text = st.text_area("Your text", 'Enter your text here')
        docs = nlp(raw_text)
        spacy_streamlit.visualize_ner(docs, labels=nlp.get_pipe('ner').labels)
if __name__ == '__main__':
    main()