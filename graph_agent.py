import os
import requests
import time
import json
import requests
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from typing import TypedDict, Optional
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

load_dotenv()
client = Groq()

# --- Configurazione ---
'''
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
MAX_RESULTS = 3       # quanti paper recuperare
MAX_RETRY = 3         # tentativi massimi in caso di 429
RETRY_DELAY = 3       # secondi di attesa iniziale (raddoppia ad ogni tentativo)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
'''


# 1. Definiamo lo Stato del Grafo
class AgentState(TypedDict):
    user_input: str
    clean_keywords: str # Parole chiave estratte dall'input dell'utente
    kb_data: Optional[str] # Evidenza scientifica trovata o None se non disponibile
    missing_info: bool # Indica se l'input dell'utente richiede dati non presenti nella KB
    final_program: Optional[str] # Contiene il programma finale generato dall'agente, se disponibile

# 2. Definiamo i Nodi del Workflow

# 3. Definiamo la Funzione Condizionale (Router)
def decidi_percorso(state: AgentState):
    if state["missing_info"]:
        return "cerca_paper"
    return "genera_scheda"

# 4. Costruzione del Grafo
workflow = StateGraph(AgentState)

# Aggiungiamo i nodi
workflow.add_node("estrai_focus", estrai_focus)
workflow.add_node("controlla_kb", controlla_kb)
workflow.add_node("cerca_paper", cerca_paper)
workflow.add_node("aggiorna_kb", aggiorna_kb)
workflow.add_node("genera_scheda", genera_scheda)

# Impostiamo il punto di ingresso
workflow.set_entry_point("estrai_focus")

# Aggiungiamo il bivio condizionale dopo il controllo KB
workflow.add_conditional_edges(
    "controlla_kb",
    decidi_percorso,
    {
        "cerca_paper": "cerca_paper",
        "genera_scheda": "genera_scheda"
    }
)

# Colleghiamo i nodi della ricerca fino alla generazione
workflow.add_edge("estrai_focus", "controlla_kb")
workflow.add_edge("cerca_paper", "aggiorna_kb")
workflow.add_edge("aggiorna_kb", "genera_scheda")
workflow.add_edge("genera_scheda", END)

# Inizializza il salvataggio in memoria
# Il memory saver permette di salvare lo stato del grafo in esecuzione, utile per debug o ripresa del flusso
memory = MemorySaver()

# Compiliamo il grafo
app = workflow.compile(checkpointer=memory)

'''
# --- GENERAZIONE E SALVATAGGIO DEL GRAFICO (VERSIONE WINDOWS) ---
print("\n--- GENERAZIONE MAPPA DEL GRAFO ---")
try:
    # Genera la stringa di testo in formato Mermaid
    codice_mermaid = app.get_graph().draw_mermaid()
    
    print("\nCopia TUTTO il blocco di testo qui sotto (comprese le linee con i trattini):\n")
    print("--------------------------------------------------")
    print(codice_mermaid)
    print("--------------------------------------------------")
    
except Exception as e:
    print(f"⚠️ Impossibile generare il codice Mermaid: {e}")
# ----------------------------------------------------------------
'''

# 5. Esecuzione di Test
if __name__ == "__main__":
    print("=== SmartFitAgent V1.0 (LangGraph Test) ===")
    
    # 1. Prepariamo l'input dell'utente
    inputs = {"user_input": "Vorrei una scheda su 3 giorni per migliorare la forza in dips e pull up zavorrate"}
    
    # 2. Definiamo la configurazione con il thread_id obbligatorio per MemorySaver
    config = {
        "configurable": {
            "thread_id": "sessione_allenamento_01"
        }
    }
    
    print("\n--- AVVIO WORKFLOW IN CORSO ---")
    
    # Variabile d'appoggio per catturare il risultato finale durante lo streaming
    risultato_finale = None

    # 3. Eseguiamo lo stream passando SEMPRE sia gli input che la config
    for output in app.stream(inputs, config=config):
        for node_name, node_state in output.items():
            print(f"✅ Nodo completato: [{node_name}]")
            
            # PROTEZIONE: Controlliamo che il node_state sia valido e non None
            if node_state and "final_program" in node_state:
                risultato_finale = node_state["final_program"]

    print("\n--- WORKFLOW COMPLETATO ---")
    print("\n=== RISPOSTA FINALE DELL'AGENTE PT ===")
    
    if risultato_finale:
        print(risultato_finale)
    else:
        # Nel caso estremo in cui get_state dia ancora problemi, usiamo questo fallback sicuro
        try:
            state_snapshot = app.get_state(config)
            print(state_snapshot.values.get("final_program", "Nessun programma trovato nello stato."))
        except Exception as e:
            print(f"Non è stato possibile recuperare lo stato: {e}")