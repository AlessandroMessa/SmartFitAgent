import os
import json
from typing import Dict
from state import AgentState

class AggiornaKb:

    def __init__(self, client_or_api_key=None):
        self.client = client_or_api_key
    
    def __call__(self, state: AgentState) -> Dict:
        print("\n[Nodo 3] -> Scrittura delle nuove scoperte scientifiche nel file JSON...")
        
        nuovi_dati = state.get("kb_data")
        file_path = "knowledge_base.json"
        
        if nuovi_dati and isinstance(nuovi_dati, dict):
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        kb = json.load(f)
                    except json.JSONDecodeError:
                        kb = {}
            else:
                kb = {}
            
            # Uniamo i vecchi dati con i nuovi
            kb.update(nuovi_dati)
            
            # Scrittura pulita su disco
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(kb, f, indent=4, ensure_ascii=False)
                
            print(f"  -> 💾 File '{file_path}' aggiornato con successo con le nuove evidenze!")
        else:
            print("  -> Avviso: nessun dato dizionario valido trovato in 'kb_data'. Salto la scrittura.")
            
        # Ritorniamo i dati mantenendo l'integrità dei tipi dello stato
        return {
            "missing_info": False,
            "kb_data": nuovi_dati  # Rimane un dizionario pulito per il Nodo 4
        }