# Relazione Esercizio 3

In questa esercitazione, abbiamo scritto ispirandoci al Text Tiling un algoritmo di Document Segmentation. I passi iniziali di tale algoritmo sono molto semplici, poiché abbiamo voluto rimanere il più possibile fedeli all'implementazione del Text Tilling di NLTK. L'algoritmo si comporta nel seguente modo:

1. Importato e inizializzato nasari all'interno di un dizionario. In una stuttura di configurazione è possibile impostare gli n primi elementi da salvare in dizionario per ogni word.

![alt text](./images/wo_metrics.png)

> **[Silhouette analysis](https://www.jeremyjordan.me/grouping-data-points-with-k-means-clustering/)**:
>
> Another more automated approach would be to build a collection of coefficient is calculated for observation, which is then averaged to determine the Silhouette score. **The coefficient combines the average within-cluster distance with average nearest-cluster distance to assign a value between -1 and 1**. A value below zero denotes that the observation is probably in the wrong cluster and a value closer to 1
> 

## Risultati
A fine esecuzione, l'algoritmo descritto, ha dato in output il seguente plot:

![alt text](./images/final_plot.png)

come si può notare, K-Means a fine iterazioni è stato in grado basandosi sulal simiarietà dellew frasi di posizionare in modo _"aprossimativamente corretto"_
i giusti breakpoint.
