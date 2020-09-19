# Relazione Esercizio 5

Questo ultimo esercizio contiene un esempio di applicazione di reti neurali
basate sul modello language model. Grazie a questa rete neruale è possibile
generare casualmente nomi a partire da un dataset di partenza.

Una volta che lo sviluppo dell'esercizio è stato terminato, ci siamo concentrati
sul sperimentare l'efficacia della rete, cambiando dataset, e aggiungendo oltre
al già presente dataset ```bands.csv``` un'altro dataset ```starwars.csv```,
contenente i nomi dei personaggi di Star Wars.   

## Osservazioni
Abbiamo effettuato alcuni esperimenti usando entrambe le funzioni ```make_name()```
(greedy) e ```make_name_beam()``` (beam search) e cambiando il dataset di
training da ```bands.csv``` a ```starwars.csv```. A parità di iterazioni,
abbiamo notato che:

- Usando il dataset ```starwars.csv``` Entrambe le funzioni hanno spesso
generato nomi che iniziano con la 'A'. Segno evidente che il dataset contiene
più entry che iniziano con tale lettera.

- Durante il testing di entrambi i dataset,La funzione greedy ```make_name()```
è riuscita a generare più nomi di ```make_name_beam()```, poiché essendo non
deterministica, calcola una probabilità diversa per ogni nome (e poi stampa
nell'output soltanto i primi tre più probabili).
- Se si usa la stessa istruzione posta a riga 101 di ```model.py``` (che riporto
di seguito) 
    
  ```python
  for i in range(3):
      make_name_beam(model, vocab, hps)  # beam search
  ```
  
  si ottengono in output 3 nomi uguali, perché il percorso che porta al primo nome
sarà sempre più probabile dei successivi due.
