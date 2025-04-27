# === Imports ===
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from langdetect import detect
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

# === Fonctions Chatbots ===

def chatbot_faq():
    print("\n🤖 Chatbot FAQ multilingue TravelGo — Tapez 'exit' pour quitter\n")
    
    df = pd.read_excel("questions_reponses_travelgo.xlsx")
    if 'Question' not in df.columns or 'Réponse' not in df.columns:
        raise ValueError("Les colonnes 'Question' ou 'Réponse' sont manquantes dans le fichier Excel.")

    questions = df['Question'].tolist()
    reponses = df['Réponse'].tolist()

    model_faq = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    question_embeddings = model_faq.encode(questions, convert_to_tensor=True)

    while True:
        user_input = input("👤 Vous : ")
        if user_input.lower() == "exit":
            print("👋 À bientôt !")
            break

        user_lang = detect(user_input)
        user_embedding = model_faq.encode(user_input, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
        best_idx = scores.argmax().item()
        best_score = scores[best_idx].item()

        if best_score > 0.5:
            reponse_fr = reponses[best_idx]
            if user_lang != 'fr':
                try:
                    reponse_traduite = GoogleTranslator(source='fr', target=user_lang).translate(reponse_fr)
                    print(f"🤖 {reponse_traduite} (traduit du français)\n")
                except Exception:
                    print(f"🤖 (FR) {reponse_fr} (⚠ erreur de traduction)\n")
            else:
                print(f"🤖 {reponse_fr}\n")
        else:
            print("🤖 Je ne trouve pas de réponse adaptée.\n")


def chatbot_falcon():
    print("\n⏳ Chargement du modèle Falcon...")
    model_id = "tiiuae/falcon-rw-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

    print("\n🤖 Chatbot généraliste Falcon prêt ! Tapez 'exit' pour quitter.\n")

    while True:
        user_input = input("👤 Vous : ")
        if user_input.lower() == "exit":
            print("👋 À bientôt !")
            break

        prompt = f"{user_input}\nRéponse :"
        response = chatbot(prompt)[0]['generated_text']
        reponse_finale = response.replace(prompt, "").strip()
        print("🤖", reponse_finale, "\n")


def chatbot_flan():
    print("\n⏳ Chargement du modèle Flan-T5...")
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    chatbot = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9
    )

    print("\n🤖 Chatbot Flan-T5 prêt ! Tapez 'exit' pour quitter.\n")

    while True:
        user_input = input("👤 Vous : ")
        if user_input.lower() == "exit":
            print("👋 À bientôt !")
            break

        prompt = f"Réponds à la question suivante : {user_input}"
        response = chatbot(prompt)[0]['generated_text']
        print("🤖", response.strip(), "\n")


# === Menu Principal ===

def main():
    while True:
        print("\n=== Menu Principal ===")
        print("1. Chatbot FAQ TravelGo (Excel)")
        print("2. Chatbot Généraliste (Falcon)")
        print("3. Chatbot Flan-T5")
        print("0. Quitter")

        choice = input("\nVotre choix : ")

        if choice == "1":
            chatbot_faq()
        elif choice == "2":
            chatbot_falcon()
        elif choice == "3":
            chatbot_flan()
        elif choice == "0":
            print("👋 Merci et à bientôt !")
            break
        else:
            print("⚠ Choix invalide, veuillez réessayer.")


if __name__ == "__main__":
    main()
