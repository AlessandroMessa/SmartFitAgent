import json
from pathlib import Path
from typing import Dict
from state import AgentState

def argomenti_correlati(chiave_db: str, testo_utente: str) -> bool:
    # Normalizziamo tutto in minuscolo
    chiave_db = chiave_db.lower().strip()
    testo_utente = testo_utente.lower().strip()
    
    # 1. Match diretto bidirezionale
    if chiave_db in testo_utente or testo_utente in chiave_db:
        return True
        
    # 2. Cluster semantici di sinonimi
    mappa_sinonimi = {
        "forza": ["forza", "zavorre", "zavorrate", "weighted", "dips", "pull up", "pull-up", "trazioni", "massimali"],
        "ipertrofia": ["massa", "ipertrofia", "muscolo", "crescita", "volume", "ipertrofico"],
        "dimagrimento": ["definizione", "cut", "dimagrire", "cardio", "deficit", "dimagrimento"]
    }
    
    for radice, sinonimi in mappa_sinonimi.items():
        chiave_nel_cluster = (radice in chiave_db) or any(s in chiave_db for s in sinonimi)
        utente_nel_cluster = (radice in testo_utente) or any(s in testo_utente for s in sinonimi)
        
        if chiave_nel_cluster and utente_nel_cluster:
            return True
            
    return False


class ControllaKb: 

    def __init__(self, client_or_api_key=None):
        self.client = client_or_api_key

    def __call__(self, state: AgentState) -> Dict:
        print("\n[Nodo 1] -> Controllo reale della Knowledge Base locale...")
        
        # 1. CORREZIONE: Clausola di Salvaguardia per evitare KeyError o stringhe vuote
        clean_keywords = state.get("clean_keywords", "")
        if not clean_keywords:
            print("  -> Avviso: Nessuna keyword definita nello stato. Salto il controllo KB.")
            return {"missing_info": True, "kb_data": None}
            
        clean_keywords_lower = clean_keywords.lower().strip()
        
        # Gestione del percorso del file (compatibile sia con main.py nella root che con sotto-cartelle)
        # Se i nodi sono dentro la cartella 'nodes/', saliamo di un livello per trovare il JSON nella root
        script_dir = Path(__file__).parent
        if script_dir.name == "nodes":
            kb_path = script_dir.parent / 'knowledge_base.json'
        else:
            kb_path = script_dir / 'knowledge_base.json'
        
        # Controlliamo se il file JSON esiste
        if not kb_path.exists():
            print("  -> Il file knowledge_base.json non esiste ancora. Deviazione su ricerca paper.")
            return {"missing_info": True, "kb_data": None}
            
        try:
            with open(kb_path, 'r', encoding='utf-8') as f:
                kb_content = json.load(f)
                
            # 2. Scansione delle chiavi nel JSON per trovare corrispondenze
            # Scegliamo di salvare direttamente il dizionario dei dettagli per coerenza di tipo nell'AgentState
            dizionario_dati_trovati = {}
            
            for argomento, dettagli in kb_content.items():
                if argomenti_correlati(argomento, clean_keywords_lower):
                    print(f"  -> Corrispondenza trovata nella KB per l'argomento: '{argomento}'")
                    # Fondiamo i dettagli trovati nel dizionario finale
                    dizionario_dati_trovati[argomento] = dettagli
            
            # 3. Gestione del risultato del bivio logico
            if dizionario_dati_trovati:
                # Trovato! Passiamo il dizionario pulito e impostiamo missing_info su False
                return {"missing_info": False, "kb_data": dizionario_dati_trovati}
            else:
                print("  -> Nessuna linea guida specifica trovata nel database locale per questa richiesta.")
                return {"missing_info": True, "kb_data": None}
                
        except Exception as e:
            print(f"  -> Errore durante la lettura della KB: {e}")
            return {"missing_info": True, "kb_data": None}