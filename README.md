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
- Estabilidade: Events Per Variable (EPV) ≥ 3 (cálculo pós-Downsampling).

#### 1.3 - Padronização

- Downsampling: Amostragem estratificada limitando bases gigantes a 10.000 instâncias.
- Cardinalidade: Variáveis numéricas com menos de 10 valores únicos reclassificadas como categóricas.
- Anonimização: Padronização dos nomes para CAT_i e NUM_j.

| Base de Dados | Repositório | Instâncias | Atributos | Numéricos (%) | % Minoria |
| :--- | :--- | :---: | :---: | :---: | :---: |
| pc4 | OpenML (AutoML) | 1.458 | 46 | 34 (73,9%) | 12,2% |
| MiniBooNE | OpenML (AutoML) | 130.064* | 50 | 50 (100,0%) | 28,1% |
| wilt | OpenML (AutoML) | 4.839 | 5 | 5 (100,0%) | 5,4% |
| churn | OpenML (AutoML) | 5.000 | 29 | 16 (55,2%) | 14,1% |
| qsar-biodeg | OpenML (AutoML) | 1.055 | 71 | 29 (40,8%) | 33,7% |
| phoneme | OpenML (AutoML) | 5.404 | 5 | 5 (100,0%) | 29,3% |
| nomao | OpenML (AutoML) | 34.465* | 177 | 74 (41,8%) | 28,6% |
| blood-transfusion | OpenML (AutoML) | 748 | 4 | 4 (100,0%) | 23,8% |
| kc1 | OpenML (AutoML) | 2.109 | 21 | 21 (100,0%) | 15,5% |
| wdbc | OpenML (CC18) | 569 | 30 | 30 (100,0%) | 37,3% |
| breast-w | OpenML (CC18) | 699 | 16 | 8 (50,0%) | 34,5% |
| diabetes | OpenML (CC18) | 768 | 8 | 8 (100,0%) | 34,9% |
| pc3 | OpenML (CC18) | 1.563 | 43 | 36 (83,7%) | 10,2% |
| kc2 | OpenML (CC18) | 522 | 27 | 20 (74,1%) | 20,5% |
| spambase | OpenML (CC18) | 4.601 | 57 | 57 (100,0%) | 39,4% |
| pc1 | OpenML (CC18) | 1.109 | 21 | 21 (100,0%) | 6,9% |
| banknote-auth | OpenML (CC18) | 1.372 | 4 | 4 (100,0%) | 44,5% |
| ilpd | OpenML (CC18) | 583 | 10 | 9 (90,0%) | 28,6% |
| higgs | OpenML (OpenML100) | 98.050* | 32 | 24 (75,0%) | 47,1% |
| steel-plates-fault | OpenML (OpenML100) | 1.941 | 34 | 24 (70,6%) | 34,7% |
| eeg-eye-state | OpenML (OpenML100) | 14.980* | 14 | 14 (100,0%) | 44,9% |
| mozilla4 | OpenML (OpenML100) | 15.545* | 5 | 4 (80,0%) | 32,9% |
| MagicTelescope | OpenML (OpenML100) | 19.020* | 10 | 10 (100,0%) | 35,2% |
| saheart | PMLB | 462 | 9 | 8 (88,9%) | 34,6% |
| profb | PMLB | 672 | 12 | 6 (50,0%) | 33,3% |
| bupa | PMLB | 345 | 5 | 5 (100,0%) | 49,0% |
| haberman | PMLB | 306 | 3 | 3 (100,0%) | 26,5% |
| clean2 | PMLB | 6.598 | 168 | 168 (100,0%) | 15,4% |
| irish | PMLB | 500 | 6 | 3 (50,0%) | 44,4% |
| tokyo1 | PMLB | 959 | 54 | 37 (68,5%) | 36,1% |
