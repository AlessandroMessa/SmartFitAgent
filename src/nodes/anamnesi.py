from typing import Dict
from state import AgentState

class Anamnesi:

    def __init__(self, client_or_api_key):
        self.client = client_or_api_key

    def __call__(self, state: AgentState) -> Dict:
        print("\n[Nodo 0] -> Pulizia dell'input ed estrazione keyword scientifiche...")
        
        prompt_sistema = """
        Sei un analista dati specializzato in scienze motorie. Il tuo compito è leggere la richiesta dell'utente ed estrarre ESCLUSIVAMENTE i termini scientifici, i pattern di movimento o gli esercizi core (es. dips, pull-ups, squat, forza, ipertrofia, rpe).
        Rimuovi saluti, preamboli, stati d'animo e parole inutili. 
        Rispondi SOLO con le parole chiave separate da virgola, senza nient'altro.
        
        Esempio: "Vorrei fare forza nelle dips con zavorra ma ho i polsi deboli"
        Output: dips, zavorra, forza, polsi
        """
        
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": state["user_input"]}
            ],
            model="llama-3.1-8b-instant", # Veloce ed economico per task di estrazione
        )
        
        keywords_pulite = response.choices[0].message.content.lower().strip()
        print(f"Keyword estratte per la ricerca: [{keywords_pulite}]")
        
        return {"clean_keywords": keywords_pulite}