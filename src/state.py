from typing import TypedDict, List, Dict, Optional

class AgentState(TypedDict):
    user_input: str
    clean_keywords: str # Parole chiave estratte dall'input dell'utente
    kb_data: Optional[str] # Evidenza scientifica trovata o None se non disponibile
    missing_info: bool # Indica se l'input dell'utente richiede dati non presenti nella KB
    final_program: Optional[str] # Contiene il programma finale generato dall'agente, se disponibile