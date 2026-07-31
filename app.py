import os
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer,util

load_dotenv()
cles = os.getenv("Groq_API_KEY")
client = Groq(api_key=cles)

reader1 = PdfReader("/home/dongift/Downloads/AFRINAI Recrutement Backend.pdf")
reader2 = PdfReader("/home/dongift/Vaults/Don-Gift/My_Road/Vocabulaire_AI_Engineer_FR_EN_Kirundi.pdf")
reader3 = PdfReader("/home/dongift/Documents/cours-ITN2026 (1).pdf")

text_complet1=""
text_complet2=""
text_complet3=""

for page in reader1.pages:
    text_complet1 +=page.extract_text()

for page in reader2.pages:
    text_complet2 +=page.extract_text()
    
for page in reader3.pages:
    text_complet3 +=page.extract_text()
total_text_complet=[text_complet1, text_complet2,text_complet3]
total_name =["AFRINAI", "Vocabulaire_AI_Engineer_FR_EN_Kirundi", "cours-ITN2026"]

def chunks(texte, taille=400):
    chunk=[]
    for i in range (0, len(texte), taille):
        chunk.append(texte[i:i+taille])
    return chunk
total_chunks=[]
for i in range (0,3):
    chunk = chunks(total_text_complet[i])
    for chun in chunk:
         total_chunks.append(
             {
        'chunk':chun,
        "source":total_name[i]
         }
         )
def trouve_chunk_pertinent(question,source_filtre=None):

    if source_filtre:
        total_chunks_filtre=[c for c in total_chunks if source_filtre==c['source']]
    else:
        total_chunks_filtre=total_chunks
    model=SentenceTransformer('all-MiniLM-L6-v2')
    text_chunk=[c['chunk'] for c in total_chunks_filtre]
    embedding_chunk = model.encode(text_chunk)
    embedding_question= model.encode(question)
    similarite=util.cos_sim(embedding_question, embedding_chunk)[0]
    indices = similarite.argsort(descending=True)
    meuilleur = [total_chunks[i] for i in indices ]

    return meuilleur

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            'role':'system',
            "content":"tu es un chatbot officiel d'une entreprise. Tu repond aux question des utilisateurs en te basant UNIQUEMENT sur les informations sur l'entreprise fournies dans le contexte ci-dessous (services,produit,stock,etc) . Si l'information demandee ne se trouve pas dans le contexte fourni, dis-le clairement et demande a utilisateur de preciser sa question, plutot d'inventer une reponse. N'ajoute jamais d'information qui ne sont pas presentes dans le contexte fourni."
        }
    ]

for message in st.session_state.messages:

    if message['role']!='system':
        with st.chat_message(message['role']):
            st.write(message['content'])

document_choisi= st.selectbox("choisir ce que tu veux qu'on discute sur",
                              ["AFRINAI", "Vocabulaire_AI_Engineer_FR_EN_Kirundi", "cours-ITN2026"])

texte= st.chat_input()

if texte:

    contexte = trouve_chunk_pertinent(texte,source_filtre= document_choisi)
    contexte_complet="\n".join([chun['chunk'] for chun in contexte])
    contenu_complet= f"Contexte {contexte_complet}\\n\n Question {texte}"

    reponse = client.chat.completions.create(
        model='openai/gpt-oss-20b',
        messages=[st.session_state.messages[0], {'role':'user','content':contenu_complet}]

    )

    result= reponse.choices[0].message.content
    st.session_state.messages.append({'role': 'user','content': texte}) 
    st.session_state.messages.append({'role':'assistant','content': result})

    st.rerun()


