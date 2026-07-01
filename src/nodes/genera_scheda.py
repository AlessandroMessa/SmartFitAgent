from state import AgentState
from typing import Dict
from state import AgentState

class GeneraScheda:
    def __init__(self, client_or_api_key):
        self.client = client_or_api_key

    def __call__(self, state: AgentState)-> Dict:
        print("\n[Nodo 4] -> Generazione della scheda con Groq usando i dati scientifici...")
        
        # Chiamata a Groq usando il contesto accumulato nello stato
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"Sei un PT d'élite. Crea una scheda basandoti SOLO su questi dati scientifici: {state['kb_data']}"
                },
                {
                    "role": "user", 
                    "content": state["user_input"]}
            ],
            model="llama-3.3-70b-versatile",
        )
        
        return {"final_program": chat_completion.choices[0].message.content}
