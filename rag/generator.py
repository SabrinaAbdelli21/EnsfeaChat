# Appel LLM + prompt
import requests
import json

def generate_answer(question, context_chunks, model_name="llama3.1:latest"):
    
    context_text = "\n\n".join([doc.page_content for doc in context_chunks])
    
    prompt = f"""Tu es un assistant. Réponds à la question en utilisant le contexte fourni. 
    Si tu ne trouves pas la réponse dans le contexte, réponds que tu ne sais pas.
    CONTEXTE: {context_text}
    QUESTION: {question}
    RÉPONSE:"""

    url = "http://host.docker.internal:11434/api/generate"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=180)
        
        if response.status_code == 404:
            return f"Erreur 404 : Le modèle '{model_name}' est introuvable. Tape 'ollama list' dans Windows pour vérifier."
            
        return response.json().get("response", "Erreur : Champ 'response' absent.")
    except Exception as e:
        return f"Erreur de connexion : {e}"