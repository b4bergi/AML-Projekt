# 1. Verschiedene MLM füllen verschiedene Lückentexte aus.

### Modelle
1. bert-base-uncased
2. facebook/bart-large
3. albert-base-v2
4. FacebookAI/roberta-base
5. google/electra-base-generator
6. deepset/gbert-base  (oder ModernGBERT, GeistBERT,...) für deutsche Texte
7. ChatGpt (zB mit Prompt: "Fülle diesen Lückentext aus: ...")
8. Humans (Wie würden Menschen den Lückentext ausfüllen)

Bei 1-5 kann der zB dieser Code zum Ausfüllen verwendet werden:
```
from transformers import pipeline

pipeline_bert = pipeline(
    task="fill-mask",
    model="bert-base-uncased",
    device=0
)
pippipeline_bert("Plants create [MASK] through a process [MASK] as photosynthesis.", top_k=5)
```

### Themengebiete
- Childrens Books
- Knowledge based texts (vielleicht Wikipedia)
- Popular book (zb Herr der Ringe: "Gandalf was a [MASK].")
- Deutscher Lückentext
- Long Text (Ein langer Text bei dem alle Lücken in einem 'Prompt' sind der dem Model gegeben wird)



### Beispielresultat
Die Modelle können dann (zb manuell) ausgewertet werden wie viele Antworten richtig sind.\
Bei 'Childrens Books' oder 'Wiki-Text' sind es zb 10 einzelne Sätze mit nur ein oder zwei Lücken.\
Bei 'Long Text' sind zb alle Lücken im selben 'Prompt'

| Modell | Childrens Books | Wiki‑Text | Deutscher Text | Long Text | .... |
|--------|--------------|-----------|----------------|-------------|----------------|
| BERT   | 6/10         | 9/10      | 7/10           | 8/10        | 5/10           |
| ALBERT | 5/10         | 8/10      | 6/10           | 7/10        | 5/10           |
| ChatGPT| 7/10         | 9/10      | 8/10           | 9/10        | 6/10           |
| ...    | 6/10         | 8/10      | 7/10           | 8/10        | 7/10           |



# 2. Lückentext Model finetunen
Usecase: Model auf nicht öffentliche Daten (zB Firmeninterne Daten) finetunen.

Gegeben ein Model (zB Bert)\
Finde einen Lückentext auf dem das Model schlecht performt. (zB Sätze wie "[MAKS] is the CEO of the company")\
Das Model auf diesen Daten finetunen. (Zb Daten, Pdfs, der imaginären Firma)\
Ziel: Das Model kann diese spezifischen Lückentexte dann ausfüllen.


# 2. Lückentext korrektur
Bestimme ob ein ausgefüllter Lückentext korrekt ist.

zB:"[MASK] is the capital of france."
Gegeben Sätze:
"Paris is the capital of france."
"Cheese is the capital of france."

Bestimme ob das erste Wort dieses Satzes korrekt ist.
In dem oberen Beispiel sollte 'Paris' eine höhere Richtigkeit haben als 'Cheese'.

#### Biespiel Implementierung:
`pippipeline_bert("[MASK] is the capital of france.", top_k=5)`
und checken ob das Wort in den Resultaten vorkommt. (Oder die Ähnlichkeit des ausgefüllten Wortes mit denen in den Resultaten checken. zB mit Sentence‑BERT, Word2Vec, ...)
