## Impacto de Técnicas de Pré-processamento de Dados no Desempenho de Algoritmos de Classificação Binária

### 1 - Seleção e Padronização de Dados
Para ranquear as técnicas de pré-processamento de forma robusta, utilizaremos o teste de Iman-Davenport seguido do post-hoc de Friedman. Como vamos comparar 14 abordagens simultâneas (13 técnicas + 1 baseline), a regra estatística (N>2k) exige que tenhamos pelo menos 28 bases de dados.

#### 1.1 -  Origem (Repositórios de Benchmark)

- OpenML (Suítes: CC18, OpenML100 e AutoML).
- PMLB (Penn Machine Learning Benchmarks).

#### 1.2 - Critérios de Inclusão e Filtragem

- Escopo: Exclusivamente problemas de Classificação Binária.
- Natureza: Datasets reais não pré-processados.
- Volume: Número de Instâncias ≥ 300.
- Proporção Numérica: ≥ 1/3 das features devem ser contínuas (cálculo pós-OHE).
- Estabilidade: Events Per Variable (EPV) ≥ 3.

#### 1.3 - Padronização

- Downsampling: Amostragem estratificada limitando bases gigantes a 10.000 instâncias.
- Tratamento de Esparsidade: Remoção de variáveis com > 50% de dados faltantes (NAs).
- Cardinalidade: Variáveis numéricas com menos de 10 valores únicos reclassificadas como categóricas.
- Anonimização: Padronização dos nomes para CAT_i e NUM_j.

| Base de Dados | Repositório | Instâncias | Atributos | Numéricos (%) | % NAs | % Minoria | EPV |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| pc4 | OpenML (AutoML) | 1.458 | 46 | 34 (73,9%) | 0,0% | 12,2% | 3,48 |
| MiniBooNE | OpenML (AutoML) | 130.064* | 50 | 50 (100,0%) | 0,0% | 28,1% | 656,98 |
| APSFailure | OpenML (AutoML) | 76.000* | 169 | 168 (99,4%) | 8,3% | 1,8% | 7,32 |
| wilt | OpenML (AutoML) | 4.839 | 5 | 5 (100,0%) | 0,0% | 5,4% | 46,98 |
| churn | OpenML (AutoML) | 5.000 | 29 | 16 (55,2%) | 0,0% | 14,1% | 21,94 |
| qsar-biodeg | OpenML (AutoML) | 1.055 | 71 | 29 (40,8%) | 0,0% | 33,7% | 4,51 |
| phoneme | OpenML (AutoML) | 5.404 | 5 | 5 (100,0%) | 0,0% | 29,3% | 285,48 |
| nomao | OpenML (AutoML) | 34.465* | 177 | 74 (41,8%) | 0,0% | 28,6% | 50,05 |
| blood-transfusion | OpenML (AutoML) | 748 | 4 | 4 (100,0%) | 0,0% | 23,8% | 40,05 |
| kc1 | OpenML (AutoML) | 2.109 | 21 | 21 (100,0%) | 0,0% | 15,5% | 13,97 |
| wdbc | OpenML (CC18) | 569 | 30 | 30 (100,0%) | 0,0% | 37,3% | 6,36 |
| breast-w | OpenML (CC18) | 699 | 16 | 8 (50,0%) | 0,2% | 34,5% | 13,56 |
| diabetes | OpenML (CC18) | 768 | 8 | 8 (100,0%) | 0,0% | 34,9% | 30,15 |
| pc3 | OpenML (CC18) | 1.563 | 43 | 36 (83,7%) | 0,0% | 10,2% | 3,35 |
| jm1 | OpenML (CC18) | 10.885* | 21 | 21 (100,0%) | 0,0% | 19,3% | 90,26 |
| kc2 | OpenML (CC18) | 522 | 27 | 20 (74,1%) | 0,0% | 20,5% | 3,57 |
| spambase | OpenML (CC18) | 4.601 | 57 | 57 (100,0%) | 0,0% | 39,4% | 28,63 |
| pc1 | OpenML (CC18) | 1.109 | 21 | 21 (100,0%) | 0,0% | 6,9% | 3,30 |
| banknote-auth | OpenML (CC18) | 1.372 | 4 | 4 (100,0%) | 0,0% | 44,5% | 137,25 |
| ilpd | OpenML (CC18) | 583 | 10 | 9 (90,0%) | 0,0% | 28,6% | 15,03 |
| higgs | OpenML (OpenML100) | 98.050* | 32 | 24 (75,0%) | 0,0% | 47,1% | 1300,02 |
| steel-plates-fault | OpenML (OpenML100) | 1.941 | 34 | 24 (70,6%) | 0,0% | 34,7% | 17,81 |
| eeg-eye-state | OpenML (OpenML100) | 14.980* | 14 | 14 (100,0%) | 0,0% | 44,9% | 432,19 |
| mozilla4 | OpenML (OpenML100) | 15.545* | 5 | 4 (80,0%) | 0,0% | 32,9% | 919,44 |
| MagicTelescope | OpenML (OpenML100) | 19.020* | 10 | 10 (100,0%) | 0,0% | 35,2% | 601,92 |
| saheart | PMLB | 462 | 9 | 8 (88,9%) | 0,0% | 34,6% | 16,00 |
| profb | PMLB | 672 | 12 | 6 (50,0%) | 0,0% | 33,3% | 16,80 |
| bupa | PMLB | 345 | 5 | 5 (100,0%) | 0,0% | 49,0% | 30,42 |
| haberman | PMLB | 306 | 3 | 3 (100,0%) | 0,0% | 26,5% | 24,30 |
| clean2 | PMLB | 6.598 | 168 | 168 (100,0%) | 0,0% | 15,4% | 5,45 |
| irish | PMLB | 500 | 6 | 3 (50,0%) | 0,0% | 44,4% | 33,30 |
| tokyo1 | PMLB | 959 | 54 | 37 (68,5%) | 0,0% | 36,1% | 5,77 |
