# Relazione Esercizio 1

Nella realizzazzione di questo esercizio abbiamo:

1. Caricato il file delle definizioni

2. Preprocessato le sentences risultanti

3. Sperimentato il calcolo di 3 diversi tipi di similarità
	- **Baseline**: Overlap sui termini delle definizioni
	- **PoS Experiment**: Overlap sul PoS Tag di ogni termine (all'interno della stessa categoria)
	- **Cosine Similarity Experiment**: Overlap sul Cosine Similarity si ogni definizione

4. Visualizzato i risultati categorizzati per concetti generici astratti/concreti e specifici astratti/concreti

## Risultati

A seguito del calcolo dei risultati, possiamo asserire che:

- In tutti e tre gli esperimenti (Baseline, PoS e Cosine Similarity Experiment), la percentuale di similarietà delle definizioni dei concetti astratti generici, tendono ad essere molto più eterogenee tra loro che nei casi concreti.

```shell script
Baseline:

         Abstract Concrete
Generic        9%      26%
Specific       9%      16%

POS Experiment:

         Abstract Concrete
Generic       65%      71%
Specific      69%      80%

Cosine Similarity Experiment:

         Abstract Concrete
Generic        7%      19%
Specific       7%      11%
```

### Appendice

#### Plot Baseline
![alt text](./images/srwo_metrics.png)

#### Plot PoS Experiment
![alt text](./images/srwo_metrics.png)

#### Plot Cosine Similarity Experiment
![alt text](./images/srwo_metrics.png)
