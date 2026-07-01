# main.py
import os
from dotenv import load_dotenv
from groq import Groq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from state import AgentState
# main.py - CORREZIONE IMPORT ESPLICITI
from nodes.anamnesi import Anamnesi
from nodes.controlla_kb import ControllaKb
from nodes.cerca_paper import CercaPaper
from nodes.aggiorna_kb import AggiornaKb
from nodes.genera_scheda import GeneraScheda 

# 1. Inizializzazione Ambiente e Client Unificato
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 2. Istanziazione dei Nodi (Dependency Injection)
# Passiamo il client Groq ai nodi che ne hanno bisogno
nodo_anamnesi = Anamnesi(client_or_api_key=groq_client)
nodo_controllo_kb = ControllaKb()
nodo_ricerca = CercaPaper(client_or_api_key=groq_client)
nodo_salvataggio = AggiornaKb()
nodo_generazione = GeneraScheda(client_or_api_key=groq_client)


# 3. Definizione delle Funzioni Router (Archi Condizionali)

def decidi_percorso_kb(state: AgentState):
    """Router del Nodo 1: Decide se andare online su PubMed o generare direttamente la scheda."""
    if state.get("missing_info", True):
        return "cerca_paper"
    return "genera_scheda"


# 4. Costruzione e Configurazione del Grafo
workflow = StateGraph(AgentState)

# Associazione dei Nodi Callable al Grafo
workflow.add_node("estrai_focus", nodo_anamnesi)
workflow.add_node("controlla_kb", nodo_controllo_kb)
workflow.add_node("cerca_paper", nodo_ricerca)
workflow.add_node("aggiorna_kb", nodo_salvataggio)
workflow.add_node("genera_scheda", nodo_generazione)

# Il Grafo si sveglia sempre eseguendo l'estrazione delle keyword
workflow.set_entry_point("estrai_focus")

# ALLINEAMENTO: Dal Nodo 0 si va direttamente e linearmente al Nodo 1
workflow.add_edge("estrai_focus", "controlla_kb")

# Bivio Condizionale in uscita dal Nodo 1 (Controllo KB)
workflow.add_conditional_edges(
    "controlla_kb",
    decidi_percorso_kb,
    {
        "cerca_paper": "cerca_paper",
        "genera_scheda": "genera_scheda"
    }
)

# Archi Lineari Standard per la pipeline di ricerca e generazione
workflow.add_edge("cerca_paper", "aggiorna_kb")
workflow.add_edge("aggiorna_kb", "genera_scheda")
workflow.add_edge("genera_scheda", END)

# Inizializzazione Checkpointer per la memoria
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# 5. Interfaccia Terminale Continuo (Ogni input genera una scheda completa)
if __name__ == "__main__":
    print("====================================================")
    print("===      SmartFitAgent V1.0 (Linear RAG Run)     ===")
    print("====================================================")
    print("Chiedi una scheda descrivendo il tuo obiettivo. Scrivi 'esci' per chiudere.\n")
    
    # Usiamo un thread_id per la sessione
    config = {
        "configurable": {
            "thread_id": "sessione_standard_pt"
        }
    }

    while True:
        try:
            user_msg = input("\nTu: ").strip()
            
            if user_msg.lower() in ["esci", "quit", "exit"]:
                print("Chiusura dell'agente. Buon allenamento!")
                break
                
            if not user_msg:
                continue
                
            # Prepariamo gli input freschi per questo specifico tentativo
            inputs = {
                "user_input": user_msg,
                "clean_keywords": "",
                "kb_data": None,
                "missing_info": False
            }
            
            print("\n[SmartFitAgent è partito! Elaborazione dell'intera pipeline...]")
            
            # Eseguiamo lo stream. Poiché il flusso è lineare, ad ogni ciclo di input 
            # l'agente farà tutto il percorso (Keyword -> KB -> eventuale PubMed -> Scheda).
            for output in app.stream(inputs, config=config):
                for node_name, node_state in output.items():
                    print(f"  -> Passo completato: [{node_name}]")
                    
                    # Quando arriviamo al nodo finale, stampiamo a schermo la scheda tecnica
                    if node_name == "genera_scheda" and "final_program" in node_state:
                        print("\n🎯 =========================================")
                        print("--- SCHEDA SCIENTIFICA GENERATA CON SUCCESSO ---")
                        print("============================================")
                        print(node_state["final_program"])
                        print("============================================\n")
                        
        except KeyboardInterrupt:
            print("\nChiusura forzata. Ci vediamo in palestra!")
            break