from typing import TypedDict, Optional
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
import json

load_dotenv()
client = Groq()

# 1. Definiamo lo Stato del Grafo
class AgentState(TypedDict):
    user_input: str
    clean_keywords: str # Parole chiave estratte dall'input dell'utente
    kb_data: Optional[str] # Evidenza scientifica trovata o None se non disponibile
    missing_info: bool # Indica se l'input dell'utente richiede dati non presenti nella KB
    final_program: Optional[str] # Contiene il programma finale generato dall'agente, se disponibile

def estrai_focus(state: AgentState) -> AgentState:
    print("\n[Nodo 0] -> Pulizia dell'input ed estrazione keyword scientifiche...")
    
    prompt_sistema = """
    Sei un analista dati specializzato in scienze motorie. Il tuo compito è leggere la richiesta dell'utente ed estrarre ESCLUSIVAMENTE i termini scientifici, i pattern di movimento o gli esercizi core (es. dips, pull-ups, squat, forza, ipertrofia, rpe).
    Rimuovi saluti, preamboli, stati d'animo e parole inutili. 
    Rispondi SOLO con le parole chiave separate da virgola, senza nient'altro.
    
    Esempio: "Vorrei fare forza nelle dips con zavorra ma ho i polsi deboli"
    Output: dips, zavorra, forza, polsi
    """
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": state["user_input"]}
        ],
        model="llama-3.1-8b-instant", # Veloce ed economico per task di estrazione
    )
    
    keywords_pulite = response.choices[0].message.content.lower().strip()
    print(f"Keyword estratte per la ricerca: [{keywords_pulite}]")
    
    return {"clean_keywords": keywords_pulite}

# 2. Definiamo i Nodi del Workflow
def controlla_kb(state: AgentState) -> AgentState:
    print("\n[Nodo 1] -> Controllo reale della Knowledge Base locale...")
    
    user_input_lower = state["user_input"].lower()
    script_dir = Path(__file__).parent
    kb_path = script_dir / 'knowledge_base.json'
    
    # 1. Controlliamo se il file JSON esiste
    if not kb_path.exists():
        print("Il file knowledge_base.json non esiste ancora. Deviazione su ricerca paper.")
        return {"missing_info": True, "kb_data": None}
        
    try:
        # 2. Leggiamo il database JSON
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb_content = json.load(f)
            
        # 3. Scansione delle chiavi nel JSON per trovare corrispondenze
        dati_trovati = []
        for argomento, dettagli in kb_content.items():
            # Se il nome dell'argomento (es. "linee_guida_forza" o "zavorre") è nell'input dell'utente
            # O se l'utente menziona parole chiave interne
            if argomenti_correlati(argomento, user_input_lower):
                print(f"Corrispondenza trovata nella KB per l'argomento: '{argomento}'")
                dati_trovati.append(f"--- {argomento.upper()} ---\n{json.dumps(dettagli, indent=2, ensure_ascii=False)}")
        
        # 4. Gestione del risultato del bivio logico
        if dati_trovati:
            # Uniamo tutte le regole scientifiche trovate in un'unica stringa nel blocco note (stato)
            contesto_kb = "\n\n".join(dati_trovati)
            return {"missing_info": False, "kb_data": contesto_kb}
        else:
            print("Nessuna linea guida specifica trovata nel database locale per questa richiesta.")
            return {"missing_info": True, "kb_data": None}
            
    except Exception as e:
        print(f"Errore durante la lettura della KB: {e}")
        return {"missing_info": True, "kb_data": None}

# Funzione di supporto interna per estendere l'intelligenza delle parole chiave
def argomenti_correlati(chiave_db: str, testo_utente: str) -> bool:
    # Mappiamo i sinonimi per rendere l'agente più elastico
    mappa_sinonimi = {
        "forza": ["forza", "zavorre", "weighted", "dips", "pull up", "pull-up", "massimali"],
        "ipertrofia": ["massa", "ipertrofia", "muscolo", "crescita", "volume"],
        "dimagrimento": ["definizione", "cut", "dimagrire", "cardio", "deficit"]
    }
    
    # Controllo diretto sul nome della chiave
    if chiave_db in testo_utente:
        return True
        
    # Controllo sui sinonimi mappati
    for radice, sinonimi in mappa_sinonimi.items():
        if radice in chiave_db:
            return any(sinonimo in testo_utente for sinonimo in sinonimi)
            
    return False

def cerca_paper(state: AgentState) -> AgentState:
    print("\n[Nodo 2] -> Dati non trovati in KB. Avvio ricerca paper scientifici online...")
    # Qui in futuro si collegherà lo strumento di ricerca (PubMed, Scholar, ecc.)
    evidenze_estratte = "Evidenza da Paper 2026: Per la forza nelle zavorre stazionare tra 3-5 min di recupero e RPE 7-9."
    return {"kb_data": evidenze_estratte}

def aggiorna_kb(state: AgentState) -> AgentState:
    print("\n[Nodo 3] -> Scrittura delle nuove scoperte scientifiche nel file JSON...")
    # Ritorniamo un dizionario valido per non mandare in crash lo stream
    return {"missing_info": False}

def genera_scheda(state: AgentState) -> AgentState:
    print("\n[Nodo 4] -> Generazione della scheda con Groq usando i dati scientifici...")
    
    # Chiamata a Groq usando il contesto accumulato nello stato
    chat_completion = client.chat.completions.create(
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
memory = MemorySaver()

# Compiliamo il grafo
app = workflow.compile()

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